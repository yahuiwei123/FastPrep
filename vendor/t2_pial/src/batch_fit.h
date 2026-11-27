#pragma once
#include <omp.h>
#include <unordered_map>

struct BatchFit {
  using Fit = std::pair<double,double>;
  std::vector<Fit> vertices;

  template<class Reader>
  void prepare(MRIS* surface, Reader read, const std::string& dir) {
    static std::unordered_map<std::string,Fit> memo;
    static size_t keyBytes = 0;
    Timer timer;
    size_t misses = 0, active = 0;
    vertices.resize(surface->nvertices);
    for(int start=0; start<surface->nvertices; start+=1024){
      if(keyBytes > 128*1024*1024 || memo.size()>131072){memo.clear();keyBytes=0;}
      std::vector<std::pair<HISTOGRAM*,Fit*>> jobs;
      std::vector<std::pair<int,const Fit*>> rows;
      for(int vertex=start;vertex<std::min(start+1024,surface->nvertices);vertex++){
        if(surface->vertices[vertex].ripflag) continue;
        auto gm=read(vertex), wm=read(vertex);
        HISTOfree(&wm);
        std::string key(reinterpret_cast<char*>(&gm->min),sizeof(gm->min));
        key.append(reinterpret_cast<char*>(&gm->max),sizeof(gm->max));
        key.append(reinterpret_cast<char*>(gm->counts),gm->nbins*sizeof(*gm->counts));
        auto entry=memo.try_emplace(std::move(key));
        Fit* result=&entry.first->second;
        rows.emplace_back(vertex,result);
        if(entry.second){
          keyBytes+=entry.first->first.size();
          jobs.emplace_back(gm,result);
        }else HISTOfree(&gm);
      }
      #pragma omp parallel for schedule(dynamic, 8) if(jobs.size()>1)
      for(int idx=0;idx<static_cast<int>(jobs.size());idx++){
        auto& job=jobs[idx];
        auto smooth=HISTOsmooth(job.first,nullptr,4);
        HISTOrobustGaussianFit(smooth,.2,&job.second->first,&job.second->second);
        HISTOfree(&smooth); HISTOfree(&job.first);
      }
      for(auto& row:rows) vertices[row.first]=*row.second;
      active+=rows.size(); misses+=jobs.size();
    }
    std::ofstream log(dir+"/batch-fit.json");
    log<<"{\"computed\":"<<misses<<",\"reused\":"<<active-misses<<",\"threads_max\":"<<omp_get_max_threads()<<",\"seconds\":"<<timer.seconds()<<"}\n";
  }
};
