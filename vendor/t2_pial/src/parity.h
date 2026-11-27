#pragma once
#include <filesystem>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <tuple>
#include "timer.h"
#include "batch_fit.h"
#include "distance_memo.h"
#include "distance_batch.h"

struct Parity {
  bool gpu = getenv("T2GPU_WORKER") != nullptr;
  bool batch = gpu && getenv("T2GPU_BATCH_FIT") && std::string(getenv("T2GPU_BATCH_FIT"))=="1" && Gdiag_no<0 && !(Gdiag & DIAG_WRITE);
  bool distanceMemo = gpu && getenv("T2GPU_DISTANCE_MEMO") && std::string(getenv("T2GPU_DISTANCE_MEMO"))=="1" && Gdiag_no<0 && !(Gdiag & DIAG_WRITE);
  bool parallelFields = distanceMemo && getenv("T2GPU_PARALLEL_FIELDS") && std::string(getenv("T2GPU_PARALLEL_FIELDS"))=="1";
  DistanceBatch distanceBatch;
  BatchFit batchFit;
  std::string dir;
  std::ofstream histout, fit, matrices;
  std::ifstream histin;
  std::map<std::tuple<int,int,int>,std::pair<double,double>> gmFits;
  Parity() {
    static int round = 0;
    dir="round-"+std::to_string(round++);
    if(!std::filesystem::create_directory(dir)) throw std::runtime_error("refuse to reuse a round");
    matrices.open(dir+"/matrices.txt"); matrices<<std::setprecision(17);
    fit.open(dir+"/fits.bin",std::ios::binary);
  }
  virtual void worker(const char* stage){
    const char* python=getenv("T2GPU_PYTHON");
    if(!python) throw std::runtime_error("T2GPU_PYTHON required");
    auto pid=fork(); if(pid<0) throw std::runtime_error("fork");
    if(pid==0){execl(python,python,getenv("T2GPU_WORKER"),stage,dir.c_str(),(char*)nullptr); _exit(127);}
    int status; if(waitpid(pid,&status,0)!=pid || status!=0) throw std::runtime_error("GPU worker failed");
  }
  void volume(MRIS* surf,const char* name,MRI* image){
    if(MRIwrite(image,(dir+"/"+name+".mgh").c_str())) throw std::runtime_error(name);
    auto map=MRIS_makeRAS2VoxelMap(image,surf); matrices<<name<<" "<<image->outside_val<<" "<<image->xsize;
    for(int row=1;row<=3;row++) for(int col=1;col<=4;col++) matrices<<" "<<map->sras2vox->rptr[row][col];
    matrices<<"\n"; MRIS_freeRAS2VoxelMap(&map);
  }
  void state(MRIS* surf,const char* file){
    std::ofstream stream(dir+"/"+file,std::ios::binary);
    for(int idx=0;idx<surf->nvertices;idx++){
      auto& vertex=surf->vertices[idx];
      float values[]={vertex.x,vertex.y,vertex.z,vertex.whitex,vertex.whitey,vertex.whitez,vertex.nx,vertex.ny,vertex.nz,vertex.pialx,vertex.pialy,vertex.pialz,(float)vertex.ripflag,vertex.targx,vertex.targy,vertex.targz,vertex.d,(float)vertex.marked,(float)vertex.marked2,vertex.val,vertex.val2};
      stream.write((char*)values,sizeof(values));
    }
    if(!stream) throw std::runtime_error(file);
  }
  void fieldVolumes(MRIS* surf,MRI* white,MRI* pial,MRI* left,MRI* right){
    volume(surf,"white",white); volume(surf,"pial",pial); volume(surf,"lh",left); volume(surf,"rh",right); matrices.close();
  }
  void inputs(MRIS* surf,double step,double maxout,double sigma,int half){
    std::ofstream cfg(dir+"/config.txt"); cfg<<std::setprecision(17)<<step<<" "<<maxout<<" "<<sigma<<" "<<half<<" "<<(surf->hemisphere==LEFT_HEMISPHERE); cfg.close();
    state(surf,"input.bin");
  }
  void prefit(MRIS* surf){
    if(gpu){worker("hist");histin.open(dir+"/gpu.hist",std::ios::binary);} else histout.open(dir+"/native.hist",std::ios::binary);
    if(batch){
      batchFit.prepare(surf,[this](int vertex){return histIn(vertex);},dir);
      if(histin.peek()!=EOF) throw std::runtime_error("extra histogram in prefit");
      histin.clear(); histin.seekg(0);
    }
  }
  void prepare(MRIS* surf,MRI* t2,MRI* seg,MRI* white,MRI* pial,MRI* left,MRI* right,double step,double maxout,double sigma,int half){
    volume(surf,"t2",t2); volume(surf,"seg",seg); fieldVolumes(surf,white,pial,left,right);
    inputs(surf,step,maxout,sigma,half); prefit(surf);
  }
  void histOut(int vno,HISTOGRAM* hist){
    float header[]={(float)vno,hist->min,hist->max,(float)hist->nbins}; histout.write((char*)header,sizeof(header));
    std::vector<uint16_t> counts(hist->nbins); for(int bin=0;bin<hist->nbins;bin++) counts[bin]=hist->counts[bin]; histout.write((char*)counts.data(),counts.size()*sizeof(uint16_t));
    if(!histout) throw std::runtime_error("hist write");
  }
  HISTOGRAM* histIn(int vno){
    float header[4]; histin.read((char*)header,sizeof(header));
    if(!histin || header[0]!=vno || header[3]<2 || header[3]>10000) throw std::runtime_error("invalid GPU histogram");
    HISTOGRAM* hist=HISTOinit(nullptr,(int)header[3],header[1],header[2]); std::vector<uint16_t> counts(hist->nbins);
    histin.read((char*)counts.data(),counts.size()*sizeof(uint16_t)); if(!histin) throw std::runtime_error("short histogram");
    for(int bin=0;bin<hist->nbins;bin++) hist->counts[bin]=counts[bin]; return hist;
  }
  void stats(int vno,double mean,double sigma,double lo,double hi,double loout,double hiout){
    double fields[]={(double)vno,mean,sigma,lo,hi,loout,hiout};fit.write((char*)fields,sizeof(fields));
  }
  void gmFit(int vertex,HISTOGRAM* hist,const MRI_REGION& region,double* mean,double* sigma){
    if(batch){std::tie(*mean,*sigma)=batchFit.vertices.at(vertex);return;}
    if(!gpu){HISTOrobustGaussianFit(hist,.2,mean,sigma);return;}
    auto key=std::make_tuple(region.x,region.y,region.z);auto known=gmFits.find(key);
    if(known!=gmFits.end()){*mean=known->second.first;*sigma=known->second.second;return;}
    HISTOrobustGaussianFit(hist,.2,mean,sigma);gmFits.emplace(key,std::make_pair(*mean,*sigma));
  }
  void finish(MRIS* surf,int& numIn,int& numOut){
    fit.close();histout.close();
    if(gpu){
      if(histin.peek()!=EOF) throw std::runtime_error("extra histogram");
      worker("search");std::ifstream stream(dir+"/gpu.fields",std::ios::binary);
      std::ifstream counts(dir+"/gpu.counts",std::ios::binary);int32_t totals[2];counts.read((char*)totals,sizeof(totals));
      if(!counts || totals[0]<0 || totals[1]<0 || (int64_t)totals[0]+totals[1]>surf->nvertices) throw std::runtime_error("invalid GPU event counts");
      numIn=totals[0];numOut=totals[1];
      for(int idx=0;idx<surf->nvertices;idx++){
        float fields[8];stream.read((char*)fields,sizeof(fields));if(!stream) throw std::runtime_error("short target file");
        auto& vertex=surf->vertices[idx];vertex.targx=fields[0];vertex.targy=fields[1];vertex.targz=fields[2];vertex.d=fields[3];vertex.marked=fields[4];vertex.marked2=fields[5];vertex.val=fields[6];vertex.val2=fields[7];
      }
      if(stream.peek()!=EOF) throw std::runtime_error("extra target");
    }
    state(surf,"output.bin");
  }
};
