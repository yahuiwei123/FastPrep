#include "geometry.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace {
void check(cudaError_t status) { if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
template<typename Type> struct Buffer {
  Type* ptr=nullptr;
  void resize(size_t count) { if(ptr) check(cudaFree(ptr)); ptr=nullptr; check(cudaMalloc(&ptr,sizeof(Type)*count)); }
  void load(const Type* host,size_t count) { check(cudaMemcpy(ptr,host,count*sizeof(Type),cudaMemcpyHostToDevice)); }
  void save(Type* host,size_t count) { check(cudaMemcpy(host,ptr,count*sizeof(Type),cudaMemcpyDeviceToHost)); }
  ~Buffer(){ if(ptr) cudaFree(ptr); }
};
Buffer<int> ids,offsets,neighbors,faceOffsets,corners;
Buffer<float> coords,distances,angles;
Buffer<float4> radial;
Buffer<FaceField> faceFields;
Buffer<VertexField> vertexFields;
int vertexCount,faceCount,edgeCount;

__device__ float3 xyz(const float* coord,int count,int id) { return make_float3(coord[id],coord[count+id],coord[2*count+id]); }
__device__ float3 sub(float3 first,float3 second) { return make_float3(first.x-second.x,first.y-second.y,first.z-second.z); }
__device__ float3 cross(float3 first,float3 second) {
  return make_float3(first.y*second.z-first.z*second.y,first.z*second.x-first.x*second.z,first.x*second.y-first.y*second.x);
}
__device__ float dot(float3 first,float3 second){ return first.x*second.x+first.y*second.y+first.z*second.z; }
__device__ float norm(float3 vec){ return sqrtf(dot(vec,vec)); }
__device__ float3 divideNorm(float3 vec){ float length=norm(vec); return length>0?make_float3(vec.x/length,vec.y/length,vec.z/length):vec; }

__global__ void normalizeRadial(const float* coord,int count,float4* radialField){
  int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=count) return;
  auto pos=xyz(coord,count,id);
  float length=sqrt(double(pos.x)*pos.x+double(pos.y)*pos.y+double(pos.z)*pos.z);
  if(length==0){radialField[id]=make_float4(1,0,0,0);return;}
  float inv=1.0f/length;
  radialField[id]=make_float4(pos.x*inv,pos.y*inv,pos.z*inv,length);
}

__global__ void edges(const float* coord,int count,const int* row,const int* column,const float4* radii,int formula,float eps,float* out){
  int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=count) return;
  auto origin=xyz(coord,count,id);
  for(int slot=row[id];slot<row[id+1];++slot){
    int neighbor=column[slot]; auto pos=xyz(coord,count,neighbor);
    if(!formula){out[slot]=norm(sub(origin,pos));continue;}
    float4 unit=radii[id]; float length=radii[neighbor].w;
    if(length<eps){out[slot]=0;continue;}
    float projection=double(unit.x)*pos.x+double(unit.y)*pos.y+double(unit.z)*pos.z;
    double cosine=projection/fmaxf(fabsf(projection),length);
    float angle=cosine<0.99f?acos(cosine):sqrt(2.0*(1.0-cosine));
    out[slot]=fabsf(angle)*unit.w;
  }
}

__device__ float approxAngle(float vertical,float horizontal,const float* table){
  float ax=fabsf(horizontal),ay=fabsf(vertical),big=fmaxf(ax,ay);
  int index=big==0?0:int((100000.0f*fminf(ax,ay))/big);
  double result=ax>=ay?table[index]:float(1.5707963267948966f-table[index]);
  if(ax!=horizontal) result=3.141592653589793f-result;
  return ay==vertical?result:-result;
}

__global__ void triangles(const float* coord,int count,const int* faceIds,int faces,const float* table,float eps,FaceField* out){
  int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=faces) return;
  float3 positions[3]; for(int corner=0;corner<3;++corner) positions[corner]=xyz(coord,count,faceIds[3*id+corner]);
  auto normal=cross(sub(positions[1],positions[0]),sub(positions[2],positions[0]));
  float length=norm(normal),scale=length<eps?1.0f:1.0f/length;
  normal=make_float3(normal.x*scale,normal.y*scale,normal.z*scale);
  FaceField field={{normal.x,normal.y,normal.z},length*.5f,{}};
  for(int corner=0;corner<3;++corner){
    auto first=sub(positions[(corner+2)%3],positions[corner]),second=sub(positions[(corner+1)%3],positions[corner]);
    field.angles[corner]=approxAngle(dot(normal,cross(second,first)),dot(first,second),table);
  }
  out[id]=field;
}

__global__ void vertexGeometry(const float* coord,int count,const int* faceIds,const int* row,const int* encoded,
                               const FaceField* faces,int step,int unitize,int divisor,VertexField* out){
  int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=count) return;
  float3 sum=make_float3(0,0,0); float area=0,initial=0;
  for(int slot=row[id];slot<row[id+1];++slot){
    int code=encoded[slot],face=code/3,corner=code%3; area+=faces[face].area;
    if(step<3) continue;
    auto pos=xyz(coord,count,id);
    auto first=sub(pos,xyz(coord,count,faceIds[face*3+(corner+2)%3]));
    auto second=sub(xyz(coord,count,faceIds[face*3+(corner+1)%3]),pos);
    initial+=norm(cross(first,second))*.5f;
    auto normal=cross(divideNorm(first),divideNorm(second));
    if(unitize) normal=divideNorm(normal);
    sum.x+=normal.x;sum.y+=normal.y;sum.z+=normal.z;
  }
  int retry=step==3 && row[id+1]>row[id] && !(norm(sum)>0);
  sum=divideNorm(sum);
  out[id]={{sum.x,sum.y,sum.z},float(double(initial)/divisor),float(double(area)/divisor),retry};
}
}

extern "C" void geomInit(int vertices,int faces,const int* faceIds,const int* row,const int* column,const int* starts,const int* incident,const float* table){
  vertexCount=vertices;faceCount=faces;edgeCount=row[vertices];
  ids.resize(3*faces); ids.load(faceIds,3*faces); offsets.resize(vertices+1); offsets.load(row,vertices+1);
  neighbors.resize(edgeCount);neighbors.load(column,edgeCount);faceOffsets.resize(vertices+1);faceOffsets.load(starts,vertices+1);
  corners.resize(starts[vertices]);corners.load(incident,starts[vertices]);angles.resize(100001);angles.load(table,100001);
  coords.resize(3*vertices);distances.resize(edgeCount);radial.resize(vertices);faceFields.resize(faces);vertexFields.resize(vertices);
}

extern "C" void geomCompute(const float* coordX,const float* coordY,const float* coordZ,int step,int formula,int unitize,int divisor,float eps,
                            float* outDist,FaceField* outFaces,VertexField* outVertices){
  const float* host[]={coordX,coordY,coordZ};
  for(int axis=0;axis<3;++axis) check(cudaMemcpy(coords.ptr+axis*vertexCount,host[axis],vertexCount*sizeof(float),cudaMemcpyHostToDevice));
  int blocks=(vertexCount+255)/256;
  if(formula) normalizeRadial<<<blocks,256>>>(coords.ptr,vertexCount,radial.ptr);
  edges<<<blocks,256>>>(coords.ptr,vertexCount,offsets.ptr,neighbors.ptr,radial.ptr,formula,eps,distances.ptr);
  if(step>=2){
    triangles<<<(faceCount+255)/256,256>>>(coords.ptr,vertexCount,ids.ptr,faceCount,angles.ptr,eps,faceFields.ptr);
    vertexGeometry<<<blocks,256>>>(coords.ptr,vertexCount,ids.ptr,faceOffsets.ptr,corners.ptr,faceFields.ptr,step,unitize,divisor,vertexFields.ptr);
  }
  check(cudaGetLastError());distances.save(outDist,edgeCount);
  if(step>=2){faceFields.save(outFaces,faceCount);vertexFields.save(outVertices,vertexCount);}
}
