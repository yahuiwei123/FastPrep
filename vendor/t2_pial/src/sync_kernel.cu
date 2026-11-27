#include <cuda_runtime.h>
#include <stdint.h>
#include "fs_tri_device.inc"

// FreeSurfer's epsilon-based triangle test is not symmetric near coplanarity.
// Planning and endpoint auditing must use exactly the same conservative predicate,
// regardless of which triangle contains the vertex currently being proposed.
__device__ bool pairIntersects(double points[6][3]){
  for(int axis=0;axis<3;++axis){double low[2],high[2];
    for(int side=0;side<2;++side){int start=3*side;low[side]=fmin(points[start][axis],fmin(points[start+1][axis],points[start+2][axis]));
      high[side]=fmax(points[start][axis],fmax(points[start+1][axis],points[start+2][axis]));}
    if(low[0]>high[1]+.002||low[1]>high[0]+.002)return false;}
  return deviceTriangle(points[0],points[1],points[2],points[3],points[4],points[5]) ||
         deviceTriangle(points[3],points[4],points[5],points[0],points[1],points[2]);
}

__global__ void pairHits(const float* xyz,const int* faces,int nfaces,const int64_t* keys,int count,unsigned char* hits){
  int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=count)return;
  int first=keys[slot]/nfaces,second=keys[slot]%nfaces;
  double points[6][3];
  for(int corner=0;corner<6;++corner){int vertex=faces[3*(corner<3?first:second)+corner%3];
    for(int axis=0;axis<3;++axis)points[corner][axis]=xyz[3*vertex+axis];}
  hits[slot]=pairIntersects(points);
}
extern "C" int classify(const float* xyz,const int* faces,int nfaces,const int64_t* keys,int count,unsigned char* hits,cudaStream_t stream){
  if(count)pairHits<<<(count+127)/128,128,0,stream>>>(xyz,faces,nfaces,keys,count,hits);
  return cudaGetLastError();
}

__global__ void orderedTrials(const float* xyz,const float* end,const int* faces,int nfaces,const int64_t* keys,int count,
                              const int* rank,const unsigned char* allowed,int* blocked){
  int work=blockIdx.x*blockDim.x+threadIdx.x,slot=work/6,corner=work%6;if(slot>=count)return;
  int ids[6];int first=keys[slot]/nfaces,second=keys[slot]%nfaces;
  for(int index=0;index<6;++index)ids[index]=faces[3*(index<3?first:second)+index%3];
  int trial=ids[corner];
  if(xyz[3*trial]==end[3*trial]&&xyz[3*trial+1]==end[3*trial+1]&&xyz[3*trial+2]==end[3*trial+2])return;
  double points[6][3];
  for(int index=0;index<6;++index){int vertex=ids[index];const float* src=(index==corner||(rank[vertex]<rank[trial]&&allowed[vertex]))?end:xyz;
    for(int axis=0;axis<3;++axis)points[index][axis]=src[3*vertex+axis];}
  if(pairIntersects(points))atomicOr(blocked+trial,1);
}
extern "C" int ordered(const float* xyz,const float* end,const int* faces,int nfaces,const int64_t* keys,int count,
                       const int* rank,const unsigned char* allowed,int* blocked,cudaStream_t stream){
  if(count)orderedTrials<<<(6*count+127)/128,128,0,stream>>>(xyz,end,faces,nfaces,keys,count,rank,allowed,blocked);
  return cudaGetLastError();
}
