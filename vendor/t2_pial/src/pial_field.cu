#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
template<class Value> struct Buffer {
  Value* data=nullptr;
  explicit Buffer(size_t count){check(cudaMalloc(&data,count*sizeof(Value)));}
  ~Buffer(){cudaFree(data);}
};

__host__ __device__ float update(float first,float second,float third){
  if(first>second){float tmp=first;first=second;second=tmp;}
  if(second>third){float tmp=second;second=third;third=tmp;}
  if(first>second){float tmp=first;first=second;second=tmp;}
  float total=first+second+third;
  float delta=total*total-3.f*(first*first+second*second+third*third-1.f);
  if(delta>=0){float sol=(total+sqrtf(delta))/3.f;if(sol+1.e-6f>=third)return sol;}
  total=first+second;delta=total*total-2.f*(first*first+second*second-1.f);
  if(delta>=0){float sol=(total+sqrtf(delta))/2.f;if(sol+1.e-6f>=second)return sol;}
  return first+1.f;
}

__global__ void prepare(const float* mask,int width,int height,int depth,float* values,uint8_t* state,int* order,int* invalid){
  int index=blockIdx.x*blockDim.x+threadIdx.x,pitch=width+2,plane=pitch*(height+2);
  if(index>=plane*(depth+2))return;
  int col=index%pitch-1,row=index/pitch%(height+2)-1,slice=index/plane-1;
  values[index]=1.e6f;state[index]=3;order[index]=-1;
  if(col<0||col>=width||row<0||row>=height||slice<0||slice>=depth)return;
  int raw=col+width*(row+height*slice),stride=width*height,rank=INT_MAX;
  float centre=mask[raw];
  if(slice && centre!=mask[raw-stride])rank=4*(raw-stride)+2;
  if(row && centre!=mask[raw-width])rank=min(rank,4*(raw-width)+1);
  if(col && centre!=mask[raw-1])rank=min(rank,4*(raw-1));
  if((slice+1<depth && centre!=mask[raw+stride])||(row+1<height && centre!=mask[raw+width])||
     (col+1<width && centre!=mask[raw+1]))rank=min(rank,4*raw+3);
  if(centre!=0.f && centre!=1.f){atomicExch(invalid,1);return;}
  values[index]=100.f;
  state[index]=rank==INT_MAX?0:2;
  if(state[index]==2){values[index]=0.5f;order[index]=rank;}
}

__global__ void relax(const float* previous,float* next,const uint8_t* state,int count,int pitch,int plane,int* changed){
  int index=blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=count)return;
  float old=previous[index],value=old;
  if(state[index]==0){float first=fminf(previous[index-1],previous[index+1]);
    float second=fminf(previous[index-pitch],previous[index+pitch]),third=fminf(previous[index-plane],previous[index+plane]);
    if(fminf(first,fminf(second,third))<old)value=fminf(old,update(first,second,third));}
  next[index]=value;unsigned votes=__ballot_sync(__activemask(),value!=old);if(votes && (threadIdx.x&31)==0)atomicExch(changed,1);
}

__global__ void finish(const float* mask,const float* padded,float* result,int width,int height,int count,float spacing){
  int index=blockIdx.x*blockDim.x+threadIdx.x;
  if(index<count){int col=index%width,row=index/width%height,slice=index/(width*height);
    int at=col+1+(width+2)*(row+1+(height+2)*(slice+1));result[index]=(mask[index]==1.f?-padded[at]:padded[at])*spacing;}
}

void march(float* values,uint8_t* state,const std::vector<int>& seeds,int pitch,int plane){
  auto later=[values](int first,int second){return values[first]>values[second];};
  std::priority_queue<int,std::vector<int>,decltype(later)> heap(later);
  int steps[]={-1,1,-pitch,pitch,-plane,plane};
  auto visit=[&](int index,bool initial){auto status=state[index];if(status>=2 || (initial && status==1))return;
    values[index]=update(std::min(values[index-1],values[index+1]),std::min(values[index-pitch],values[index+pitch]),
                         std::min(values[index-plane],values[index+plane]));
    if(!status){state[index]=1;heap.push(index);}};
  for(int seed:seeds)for(int step:steps)visit(seed+step,true);
  while(!heap.empty()){
    int top=heap.top();if(values[top]>=100.f){while(!heap.empty()){values[heap.top()]=100.f;heap.pop();}break;}
    heap.pop();state[top]=2;for(int step:steps)visit(top+step,false);
  }
}
#ifdef PIAL_DEVICE_ORDER
#include "pial_device_order.cuh"
#endif
}

extern "C" int pial_field(const float* mask,float* output,int width,int height,int depth,float spacing,int ordered,int* passes,char* error,size_t length){
  try{
    if(width<2||height<2||depth<2||width>4096||height>4096||depth>4096||
       int64_t(width+2)*(height+2)*(depth+2)>INT_MAX/4||!std::isfinite(spacing)||spacing<=0)
      throw std::runtime_error("unsupported pial grid");
    int pitch=width+2,plane=pitch*(height+2),count=plane*(depth+2),size=width*height*depth;
    Buffer<float> raw(size),values(count),result(size);Buffer<uint8_t> state(count);Buffer<int> order(count);
    check(cudaMemcpy(raw.data,mask,size*sizeof(float),cudaMemcpyHostToDevice));
    Buffer<int> invalid(1);int bad=0;check(cudaMemset(invalid.data,0,sizeof(int)));
    prepare<<<(count+255)/256,256>>>(raw.data,width,height,depth,values.data,state.data,order.data,invalid.data);check(cudaGetLastError());
    check(cudaMemcpy(&bad,invalid.data,sizeof(int),cudaMemcpyDeviceToHost));if(bad)throw std::runtime_error("expected binary float pial mask");
    *passes=0;
#ifdef PIAL_DEVICE_ORDER
    if(ordered==2){
      Buffer<int> rows(size_t(4)*size),counts(height*depth),heap(size_t(2)*size);
      seedRows<<<(height*depth+127)/128,128>>>(order.data,rows.data,counts.data,width,height,depth);check(cudaGetLastError());
      serialMarch<<<2,1>>>(raw.data,rows.data,counts.data,heap.data,values.data,state.data,width,height,depth);
      check(cudaGetLastError());check(cudaDeviceSynchronize());
    }else
#endif
    if(ordered==1){
      std::vector<int> ranks(count),seeds[2];std::vector<float> field(count);std::vector<uint8_t> status(count);
      check(cudaMemcpy(status.data(),state.data,count,cudaMemcpyDeviceToHost));
      check(cudaMemcpy(ranks.data(),order.data,count*sizeof(int),cudaMemcpyDeviceToHost));
      check(cudaMemcpy(field.data(),values.data,count*sizeof(float),cudaMemcpyDeviceToHost));
      for(int index=0;index<count;++index)if(ranks[index]>=0){
        int at=index%pitch-1+width*(index/pitch%(height+2)-1+height*(index/plane-1));seeds[int(mask[at])].push_back(index);}
      for(auto& side:seeds){std::sort(side.begin(),side.end(),[&](int first,int second){return ranks[first]<ranks[second];});
        march(field.data(),status.data(),side,pitch,plane);}
      check(cudaMemcpy(values.data,field.data(),count*sizeof(float),cudaMemcpyHostToDevice));
    }else if(ordered==0){
      Buffer<float> next(count);Buffer<int> changed(1);int active=1;
      while(active && *passes<2*(width+height+depth)){
        check(cudaMemset(changed.data,0,sizeof(int)));relax<<<(count+255)/256,256>>>(values.data,next.data,state.data,count,pitch,plane,changed.data);
        check(cudaGetLastError());std::swap(next.data,values.data);++*passes;
        check(cudaMemcpy(&active,changed.data,sizeof(int),cudaMemcpyDeviceToHost));}
      if(active)throw std::runtime_error("parallel relaxation did not converge");
    }else throw std::runtime_error("unsupported pial backend");
    finish<<<(size+255)/256,256>>>(raw.data,values.data,result.data,width,height,size,spacing);check(cudaGetLastError());
    check(cudaMemcpy(output,result.data,size*sizeof(float),cudaMemcpyDeviceToHost));return 0;
  }catch(const std::exception& failure){snprintf(error,length,"%s",failure.what());return 1;}
}
