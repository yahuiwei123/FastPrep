#pragma once
#include "parity.h"
#include <chrono>
#include <sys/socket.h>

class TargetSession {
  int channel=-1;
  pid_t pid=-1;
  void closeAndWait(bool check){
    if(channel>=0){close(channel);channel=-1;}
    if(pid<0)return;
    int status=0;pid_t waited;
    do {waited=waitpid(pid,&status,0);} while(waited<0 && errno==EINTR);
    pid=-1;
    if(check && (waited<0 || !WIFEXITED(status) || WEXITSTATUS(status)!=0))throw std::runtime_error("target worker failed");
  }
  void acknowledge(char expected){
    char token=0;ssize_t count;
    do {count=recv(channel,&token,1,0);} while(count<0 && errno==EINTR);
    if(count!=1 || token!=expected)throw std::runtime_error("target worker failed before stage acknowledgement");
  }
public:
  ~TargetSession(){closeAndWait(false);}
  TargetSession()=default;
  TargetSession(const TargetSession&)=delete;
  TargetSession& operator=(const TargetSession&)=delete;
  void histogram(const std::string& dir){
    const char* python=getenv("T2GPU_PYTHON");const char* worker=getenv("T2GPU_WORKER");
    if(pid>=0 || !python || !worker)throw std::runtime_error("invalid target session");
    auto script=std::filesystem::path(worker).parent_path()/"target_session.py";
    char* args[]={const_cast<char*>(python),const_cast<char*>(script.c_str()),const_cast<char*>(dir.c_str()),nullptr};
    int pair[2];if(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,pair))throw std::runtime_error("socketpair");
    posix_spawn_file_actions_t actions;
    int error=posix_spawn_file_actions_init(&actions);
    if(!error){
      error=posix_spawn_file_actions_adddup2(&actions,pair[1],STDIN_FILENO);
      pid_t child=-1;
      if(!error)error=posix_spawn(&child,python,&actions,nullptr,args,environ);
      if(!error)pid=child;
      posix_spawn_file_actions_destroy(&actions);
    }
    close(pair[1]);channel=pair[0];
    if(error)throw std::runtime_error("cannot spawn target session");
    acknowledge('H');
  }
  void search(){
    ssize_t count;do {count=send(channel,"S",1,MSG_NOSIGNAL);} while(count<0 && errno==EINTR);
    if(count!=1)throw std::runtime_error("cannot release target search");
    acknowledge('S');closeAndWait(true);
  }
};

struct TargetTrial : Parity {
  TargetSession session;
  using Clock=std::chrono::steady_clock;
  Clock::time_point origin=Clock::now();
  double now() const {return std::chrono::duration<double>(Clock::now()-origin).count();}
  void worker(const char* stage) override {
    if(!std::strcmp(stage,"hist"))session.histogram(dir);
    else if(!std::strcmp(stage,"search"))session.search();
    else throw std::runtime_error("unknown target stage");
  }
  void prepare(MRIS* surf,MRI* t2,MRI* seg,MRI* white,MRI* pial,MRI* left,MRI* right,double step,double maxout,double sigma,int half){
    if(!batch || !parallelFields || !compactHeapBuild())throw std::runtime_error("requires compact batched T2 driver");
    int budget=omp_get_max_threads(),fitThreads=budget,fieldThreads=std::min(4,budget);
    bool overlap=false;
#ifdef T2GPU_TARGET_OVERLAP
    for(int slot=0;slot<3;++slot){
      const auto& job=distanceBatch.jobs[slot];
      if(!DistanceMemo::entry(slot).matches(job.mask,job.field,job.label))break;
      if(slot==2)fieldThreads=1;
    }
    overlap=budget>fieldThreads;
    if(overlap)fitThreads=budget-fieldThreads;
#endif
    double fieldStart,fieldEnd,fitStart,fitEnd;
    auto fields=[&]{fieldStart=now();distanceBatch.finish(dir,overlap?fieldThreads:0);fieldEnd=now();};
    auto fit=[&]{fitStart=now();prefit(surf);fitEnd=now();};
    if(!overlap)fields();
    volume(surf,"t2",t2);volume(surf,"seg",seg);matrices.flush();
    inputs(surf,step,maxout,sigma,half);
    if(overlap){
      auto future=std::async(std::launch::async,[&]{omp_set_num_threads(fitThreads);fit();});
      fields();future.get();
    }else fit();
    fieldVolumes(surf,white,pial,left,right);
    std::ofstream log(dir+"/target-schedule.json");
    log<<std::setprecision(12)<<"{\"overlap\":"<<(overlap?"true":"false")<<",\"budget\":"<<budget
       <<",\"field_slots\":"<<fieldThreads<<",\"fit_threads\":"<<fitThreads<<",\"fields\":["<<fieldStart<<','<<fieldEnd
       <<"],\"hist_fit\":["<<fitStart<<','<<fitEnd<<"]}\n";
  }
};
