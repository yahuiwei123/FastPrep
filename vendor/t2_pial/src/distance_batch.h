#pragma once
#include <cerrno>
#include <filesystem>
#include <future>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include "distance_memo.h"
#include "lean_distance.h"
#ifdef T2GPU_ORDERED_PIAL
extern "C" int pial_field(const float*,float*,int,int,int,float,int,int*,char*,size_t);
#endif

struct DistanceBatch {
  struct Job { MRI* mask; MRI* field; int label; };
  std::array<Job,4> jobs{};
  static constexpr std::array<const char*,4> names={"lh","rh","white","pial"};

#ifdef T2GPU_ORDERED_PIAL
  static void orderedPial(MRI* mask,MRI* field,int label){
    if(label!=1 || mask->type!=MRI_FLOAT || !DistanceMemo::eligible(mask,field) || mask->width!=field->width ||
       mask->height!=field->height || mask->depth!=field->depth || mask->xsize!=mask->ysize || mask->xsize!=mask->zsize)
      throw std::runtime_error("unsupported ordered pial input");
    int passes;char error[512]{};
#ifdef T2GPU_PARALLEL_PIAL
    constexpr int mode=0;
#elif defined(T2GPU_DEVICE_PIAL)
    constexpr int mode=2;
#else
    constexpr int mode=1;
#endif
    if(pial_field(static_cast<const float*>(mask->chunk),static_cast<float*>(field->chunk),mask->width,mask->height,
                  mask->depth,mask->xsize,mode,&passes,error,sizeof(error)))throw std::runtime_error(error);
    field->outside_val=100;
  }
#endif

  static void child(const std::string& input,const std::string& output,int label,MRI* field,bool lean){
    auto exe=std::filesystem::read_symlink("/proc/self/exe").parent_path()/"distance-field";
    std::string number=std::to_string(label);
    std::string mode=lean?"lean":"native";
    char* args[]={const_cast<char*>(exe.c_str()),const_cast<char*>(input.c_str()),const_cast<char*>(output.c_str()),number.data(),mode.data(),nullptr};
    pid_t pid; int status;
    if(posix_spawn(&pid,exe.c_str(),nullptr,nullptr,args,environ)!=0) throw std::runtime_error("cannot spawn "+exe.string());
    pid_t waited; do { waited=waitpid(pid,&status,0); } while(waited<0 && errno==EINTR);
    if(waited!=pid || !WIFEXITED(status) || WEXITSTATUS(status)!=0) throw std::runtime_error("distance worker failed: "+output);
    std::ifstream result(output,std::ios::binary);
    result.read(reinterpret_cast<char*>(&field->outside_val),sizeof(field->outside_val));
    result.read(static_cast<char*>(field->chunk),field->bytes_total);
    if(!result || result.peek()!=EOF || field->outside_val!=100) throw std::runtime_error("invalid distance field: "+output);
  }

  void finish(const std::string& dir,int nativeLimit=0){
    Timer elapsed;
    std::vector<int> pending;
    int threads=std::min(4,omp_get_max_threads());
    struct RestoreThreads {
      int previous;
      ~RestoreThreads(){if(previous)omp_set_num_threads(previous);}
    } restore{nativeLimit?omp_get_max_threads():0};
    if(nativeLimit)omp_set_num_threads(nativeLimit);
    bool supported=true;
    for(int slot=0;slot<4;slot++){
      auto [mask,field,label]=jobs[slot];
      supported=supported && DistanceMemo::eligible(mask,field) && (mask->type==MRI_UCHAR || mask->type==MRI_SHORT || mask->type==MRI_FLOAT);
      if(slot==3 || !DistanceMemo::entry(slot).matches(mask,field,label)) pending.push_back(slot);
    }
    bool parallel=supported && threads>1 && pending.size()>1;
    bool lean=supported && getenv("T2GPU_LEAN_MARCH") && std::string(getenv("T2GPU_LEAN_MARCH"))=="1";
    if(!parallel){
      auto transform=[lean](MRI* src,MRI* dst,int label){
        if(lean) leanSignedField(src,dst,label);
        else DistanceMemo::native(src,dst,label);
      };
      for(int slot=0;slot<4;slot++){
#ifdef T2GPU_ORDERED_PIAL
        if(slot==3){
          const char* capture=getenv("T2GPU_CAPTURE_WARM_PIAL");
          if(capture && !strcmp(capture,"1") && MRIwrite(jobs[slot].mask,(dir+"/mask-pial.mgh").c_str()))throw std::runtime_error("warm mask capture failed");
          DistanceMemo::run(dir,slot,jobs[slot].mask,jobs[slot].field,jobs[slot].label,orderedPial);continue;}
#endif
        DistanceMemo::run(dir,slot,jobs[slot].mask,jobs[slot].field,jobs[slot].label,transform);
      }
    }else{
      std::map<MRI*,std::string> files;
      for(int slot:pending){
        auto mask=jobs[slot].mask;
        if(!files.count(mask)){
          auto file=dir+"/mask-"+names[slot]+".mgh";
          if(MRIwrite(mask,file.c_str())) throw std::runtime_error(file);
          files.emplace(mask,file);
        }
      }
      std::array<std::string,4> outcomes;
      std::array<double,4> times{};
      for(int start=0;start<4;start+=threads){
        std::vector<std::future<void>> workers;
        for(int slot=start;slot<std::min(start+threads,4);slot++) workers.push_back(std::async(std::launch::async,[&,slot]{
          Timer timer; auto [mask,field,label]=jobs[slot];
          auto transform=[&](MRI* src,MRI* dst,int key){child(std::as_const(files).at(src),dir+"/field-"+names[slot]+".raw",key,dst,lean);};
          if(slot==3){
#ifdef T2GPU_ORDERED_PIAL
            orderedPial(mask,field,label);
#else
            transform(mask,field,label);
#endif
            outcomes[slot]="recompute";}
          else outcomes[slot]=DistanceMemo::entry(slot).apply(mask,field,label,transform);
          times[slot]=timer.seconds();
        }));
        for(auto& worker:workers) worker.get();
      }
      std::ofstream log(dir+"/distance-cache.tsv",std::ios::app);
      for(int slot=0;slot<4;slot++) log<<names[slot]<<'\t'<<outcomes[slot]<<'\t'<<times[slot]<<'\n';
    }
    int children=parallel?int(pending.size()):0;auto log=std::ofstream(dir+"/distance-batch.json");
#ifdef T2GPU_ORDERED_PIAL
    if(parallel)--children;
#ifdef T2GPU_PARALLEL_PIAL
    log<<"{\"pial_backend\":\"cuda-parallel-relaxation\",";
#elif defined(T2GPU_DEVICE_PIAL)
    log<<"{\"pial_backend\":\"cuda-device-order\",";
#else
    log<<"{\"pial_backend\":\"cuda-seeds-host-order\",";
#endif
#else
    log<<'{';
#endif
    log<<"\"children\":"<<children<<",\"limit\":"<<threads<<",\"lean\":"<<(lean?"true":"false")<<",\"compact_heap\":"<<(lean && compactHeapBuild()?"true":"false")<<",\"wall_seconds\":"<<elapsed.seconds()<<"}\n";
  }
};
