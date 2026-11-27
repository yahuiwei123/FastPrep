#include "mrisurf.h"
#include "mrishash.h"
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <spawn.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {
void writeOneRing(const MRIS* surface){
  static_assert(sizeof(int)==4);
  std::ofstream offsets("sync-moves/offsets.i32",std::ios::binary),neighbors("sync-moves/neighbors.i32",std::ios::binary);
  int offset=0;
  for(int vertex=0;vertex<surface->nvertices;++vertex){
    const auto& ring=surface->vertices_topology[vertex];
    if(ring.vnum<0 || (ring.vnum && !ring.v) || ring.vnum>std::numeric_limits<int>::max()-offset)
      throw std::runtime_error("invalid MRIS one-ring");
    offsets.write(reinterpret_cast<const char*>(&offset),sizeof(offset));
    if(ring.vnum)neighbors.write(reinterpret_cast<const char*>(ring.v),sizeof(int)*ring.vnum);
    offset+=ring.vnum;
  }
  offsets.write(reinterpret_cast<const char*>(&offset),sizeof(offset));
  offsets.close();neighbors.close();if(!offsets||!neighbors)throw std::runtime_error("one-ring write failed");
}

struct Worker {
  int fd=-1,attempt=0;pid_t pid=-1;
  const MRIS* mesh=nullptr;int vertexCount=0,faceCount=0;
  void start(MRIS* surface){
    const auto& geometry=surface->vg;
    if(!geometry.valid || !(geometry.xsize>0 && geometry.ysize>0 && geometry.zsize>0 &&
                            geometry.xsize+geometry.ysize+geometry.zsize<=3.0f))
      throw std::runtime_error("unsupported voxel geometry for the 0.01 mm neighbor rule");
    if(!std::filesystem::create_directory("sync-moves"))throw std::runtime_error("refuse to reuse sync-moves");
    std::ofstream faces("sync-moves/faces.i32",std::ios::binary);
    for(int face=0;face<surface->nfaces;++face)for(int corner=0;corner<3;++corner){int vertex=surface->faces[face].v[corner];faces.write(reinterpret_cast<char*>(&vertex),sizeof(int));}
    faces.close();if(!faces)throw std::runtime_error("invalid faces");
    writeOneRing(surface);
    mesh=surface;vertexCount=surface->nvertices;faceCount=surface->nfaces;
    char* argv[]={getenv("T2GPU_PYTHON"),getenv("SYNC_WORKER"),getenv("SYNC_LIBRARY"),nullptr};
    if(!argv[0]||!argv[1]||!argv[2])throw std::runtime_error("sync worker unconfigured");
    int pair[2];if(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,pair))throw std::runtime_error("socketpair");
    posix_spawn_file_actions_t actions;if(posix_spawn_file_actions_init(&actions))throw std::runtime_error("file actions");
    if(posix_spawn_file_actions_adddup2(&actions,pair[1],STDIN_FILENO)||posix_spawn(&pid,argv[0],&actions,nullptr,argv,environ))throw std::runtime_error("spawn");
    posix_spawn_file_actions_destroy(&actions);close(pair[1]);fd=pair[0];
  }
  ~Worker(){if(fd>=0)close(fd);if(pid>0){int status;while(waitpid(pid,&status,0)<0&&errno==EINTR){}}}
  void notify(){char answer=0;ssize_t count;
    do{count=send(fd,"M",1,MSG_NOSIGNAL);}while(count<0&&errno==EINTR);
    if(count!=1)throw std::runtime_error("worker send");
    do{count=recv(fd,&answer,1,0);}while(count<0&&errno==EINTR);
    if(count!=1||answer!='M')throw std::runtime_error("GPU move failed");}
};
}

double mrisAsynchronousTimeStep(MRIS* surface,float momentum,float dt,MHT* table,float limit){
  static Worker worker;
  if(!table||table->which()!=CURRENT_VERTICES||table->fno_usage()!=MHTFNO_FACE||momentum!=0||dt<=0||limit<=0)
    throw std::runtime_error("unsupported T2 movement");
  if(worker.fd<0)worker.start(surface);
  if(surface!=worker.mesh || surface->nvertices!=worker.vertexCount || surface->nfaces!=worker.faceCount)
    throw std::runtime_error("movement mesh changed after one-ring upload");
  const auto dir="sync-moves/step-"+std::to_string(worker.attempt++);
  if(!std::filesystem::create_directory(dir))throw std::runtime_error("existing step");
  std::ofstream config(dir+"/config.txt");config<<std::setprecision(12)<<surface->nvertices<<' '<<dt<<' '<<momentum<<' '<<limit;config.close();
  std::ofstream stream(dir+"/input.f32",std::ios::binary);
  for(int vertex=0;vertex<surface->nvertices;++vertex){auto& point=surface->vertices[vertex];
    float state[]={point.x,point.y,point.z,point.dx,point.dy,point.dz,point.odx,point.ody,point.odz,float(point.ripflag)};
    stream.write(reinterpret_cast<char*>(state),sizeof(state));}
  stream.close();if(!stream||!config)throw std::runtime_error("movement input write");worker.notify();
  std::ifstream output(dir+"/output.f32",std::ios::binary);std::vector<float> out(7*size_t(surface->nvertices));
  if(!output.read(reinterpret_cast<char*>(out.data()),out.size()*sizeof(float))||output.peek()!=EOF)throw std::runtime_error("invalid GPU response");
  MRISfreeDistsButNotOrig(surface);
  for(int vertex=0;vertex<surface->nvertices;++vertex){auto& point=surface->vertices[vertex];auto values=&out[7*vertex];
    for(int component=0;component<7;++component)if(!std::isfinite(values[component]))throw std::runtime_error("nonfinite GPU move");
    if(point.ripflag){if(values[0]!=point.x||values[1]!=point.y||values[2]!=point.z)throw std::runtime_error("frozen vertex moved");continue;}
    MRISsetXYZ(surface,vertex,values[0],values[1],values[2]);
    point.dx=point.odx=values[3];point.dy=point.ody=values[4];point.dz=point.odz=values[5];point.cropped=values[6]?point.cropped+1:0;}
  return dt;
}
