#pragma once
#include <array>
#include <cstring>
#include <functional>

struct DistanceMemo {
  using Grid = std::array<double,33>;
  Grid sourceGrid{}, targetGrid{};
  int label = -1;
  double outside = 0;
  std::vector<unsigned char> mask, values;

  static Grid grid(const MRI* image){
    return {double(image->width),double(image->height),double(image->depth),double(image->nframes),double(image->type),
      double(image->ras_good_flag),image->xsize,image->ysize,image->zsize,image->x_r,image->x_a,image->x_s,
      image->y_r,image->y_a,image->y_s,image->z_r,image->z_a,image->z_s,image->c_r,image->c_a,image->c_s,
      image->s_r,image->s_a,image->s_s,image->xstart,image->ystart,image->zstart,image->xend,image->yend,image->zend,
      image->outside_val,double(image->imnr0),double(image->imnr1)};
  }

  static bool contiguous(const MRI* image){
    return image->nframes==1 && image->ischunked && image->chunk && image->bytes_total>0 &&
      image->bytes_total==size_t(image->width)*image->height*image->depth*image->bytes_per_vox;
  }

  static bool eligible(const MRI* source,const MRI* target){
    return contiguous(source) && contiguous(target) && target->type==MRI_FLOAT && source->bytes_total+target->bytes_total<=160*1024*1024;
  }
  bool matches(const MRI* source,const MRI* target,int requestedLabel) const {
    return eligible(source,target) && label==requestedLabel && sourceGrid==grid(source) && targetGrid==grid(target) &&
       mask.size()==source->bytes_total && values.size()==target->bytes_total && !std::memcmp(mask.data(),source->chunk,mask.size());
  }
  static void native(MRI* source,MRI* target,int label){ MRIdistanceTransform(source,target,label,100,DTRANS_MODE_SIGNED,nullptr); }
  const char* apply(MRI* source, MRI* target, int requestedLabel,const std::function<void(MRI*,MRI*,int)>& compute=native){
    if(!eligible(source,target)){
      compute(source,target,requestedLabel);
      return "bypass";
    }
    auto srcGrid=grid(source), dstGrid=grid(target);
    if(matches(source,target,requestedLabel)){
      std::memcpy(target->chunk,values.data(),values.size()); target->outside_val=outside;
      return "hit";
    }
    compute(source,target,requestedLabel);
    sourceGrid=srcGrid; targetGrid=dstGrid; label=requestedLabel; outside=target->outside_val;
    mask.resize(source->bytes_total); std::memcpy(mask.data(),source->chunk,mask.size());
    values.resize(target->bytes_total); std::memcpy(values.data(),target->chunk,values.size());
    return "miss";
  }

  static DistanceMemo& entry(int slot){
    static std::array<DistanceMemo,3> cache;
    return cache.at(slot);
  }
  static void run(const std::string& dir,int slot,MRI* source,MRI* target,int label,const std::function<void(MRI*,MRI*,int)>& compute=native){
    static const std::array<const char*,4> names={"lh","rh","white","pial"};
    Timer timer;
    const char* status="recompute";
    if(slot==3) compute(source,target,label);
    else status=entry(slot).apply(source,target,label,compute);
    std::ofstream(dir+"/distance-cache.tsv",std::ios::app)<<names.at(slot)<<'\t'<<status<<'\t'<<timer.seconds()<<'\n';
  }
};
