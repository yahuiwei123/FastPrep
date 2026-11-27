#define COMPILING_MRISURF_METRIC_PROPERTIES
#define COMPILING_MRISURF_TOPOLOGY_FRIEND_CHECKED
#include "mrisurf_metricProperties.h"
#include "mrisurf_MRIS.h"
#include "mrisurf_MRIS_MP.h"
#include "mrisurf.h"
#include "romp_support.h"
#include "fnv_hash.h"
#include "utils.h"
#include "geometry.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
extern int fix_vertex_area;
float* mrisStealDistStore(MRIS*,int,int);
static const int debugNonDeterminism=0;
static int int_compare(const void* first,const void* second){return *static_cast<const int*>(first)-*static_cast<const int*>(second);}
#include "official_parts.inc"

int mrismp_OrientSurface(MRIS_MP*);

namespace {
struct Resident {
  std::vector<int> ids,row,neighbors,starts,incident;
  std::vector<float> dist;
  std::vector<FaceField> faces;
  std::vector<VertexField> vertices;
  bool matches(MRIS_MP* mesh) {
    if(int(vertices.size())!=mesh->nvertices || int(faces.size())!=mesh->nfaces) return false;
    for(int face=0;face<mesh->nfaces;++face)
      if(std::memcmp(&ids[3*face],mesh->faces_topology[face].v,3*sizeof(int))) return false;
    for(int id=0;id<mesh->nvertices;++id){
      auto& entry=mesh->vertices_topology[id];
      if(row[id+1]-row[id]!=entry.vtotal || starts[id+1]-starts[id]!=entry.num ||
         std::memcmp(neighbors.data()+row[id],entry.v,entry.vtotal*sizeof(int))) return false;
      for(int slot=0;slot<entry.num;++slot) if(incident[starts[id]+slot]!=3*entry.f[slot]+entry.n[slot]) return false;
    }
    return true;
  }
  void bind(MRIS_MP* mesh) {
    if(std::any_of(mesh->v_ripflag,mesh->v_ripflag+mesh->nvertices,[](char flag){return flag!=0;}) ||
       std::any_of(mesh->f_ripflag,mesh->f_ripflag+mesh->nfaces,[](char flag){return flag!=0;})) throw std::runtime_error("ripped mesh not qualified");
    if(matches(mesh)) return;
    ids.clear();neighbors.clear();incident.clear();row={0};starts={0};
    for(int face=0;face<mesh->nfaces;++face){
      auto corners=mesh->faces_topology[face].v;
      for(int slot=0;slot<3;++slot){if(corners[slot]<0 || corners[slot]>=mesh->nvertices) throw std::runtime_error("invalid face");ids.push_back(corners[slot]);}
    }
    for(int id=0;id<mesh->nvertices;++id){
      auto& entry=mesh->vertices_topology[id];
      for(int slot=0;slot<entry.vtotal;++slot){int other=entry.v[slot];if(other<0 || other>=mesh->nvertices) throw std::runtime_error("invalid edge");neighbors.push_back(other);}
      for(int slot=0;slot<entry.num;++slot){int face=entry.f[slot],corner=entry.n[slot];
        if(face<0 || face>=mesh->nfaces || corner<0 || corner>=3 || ids[3*face+corner]!=id) throw std::runtime_error("invalid incident face");
        incident.push_back(3*face+corner);
      }
      row.push_back(neighbors.size());starts.push_back(incident.size());
    }
    std::vector<float> table(100001);
    for(int index=0;index<=100000;++index) table[index]=std::atan2(double(index),100000.0);
    geomInit(mesh->nvertices,mesh->nfaces,ids.data(),row.data(),neighbors.data(),starts.data(),incident.data(),table.data());
    faces.resize(mesh->nfaces);vertices.resize(mesh->nvertices);dist.resize(neighbors.size());
    std::fprintf(stderr,"GEOM_BIND %d %d %zu\n",mesh->nvertices,mesh->nfaces,neighbors.size());
  }
};

void equal(float reference,float candidate,const char* field,int index){
  if(reference!=candidate || !std::isfinite(candidate)){
    std::fprintf(stderr,"GEOM_DIFFERENCE %s %d reference=%a candidate=%a\n",field,index,double(reference),double(candidate));
    throw std::runtime_error("geometry check failed");
  }
}

float sumFaces(MRIS_MP* mesh){
  double total=0;
#define ROMP_VARIABLE face
#define ROMP_LO 0
#define ROMP_HI mesh->nfaces
#define ROMP_SUMREDUCTION0 total
#define ROMP_FOR_LEVEL ROMP_level_shown_reproducible
#ifdef ROMP_SUPPORT_ENABLED
  const int romp_for_line=__LINE__;
#endif
#include "romp_for_begin.h"
  ROMP_for_begin
    ROMP_PARTIALSUM(0)+=mesh->f_area[face];
#include "romp_for_end.h"
  return total;
}
}

void MRIScomputeMetricProperties(MRIS_MP* mesh){
  static const int step=[](){const char* value=std::getenv("GEOM_STEP");return value?std::atoi(value):0;}();
  static const bool audit=std::getenv("GEOM_CHECK") && std::string(std::getenv("GEOM_CHECK"))=="1";
  if(step<1 || step>3 || std::getenv("FREESURFER_fastApproxAtan2f_use_atanf2")) throw std::runtime_error("unsupported geometry mode");
  static Resident cache;
  cache.bind(mesh);
  if(step<3) MRISMP_computeNormals(mesh,false);
  auto calculate=[&](){ geomCompute(mesh->v_x,mesh->v_y,mesh->v_z,step,MRIS_Status_distanceFormula(mesh->status),UnitizeNormalFace,
                          fix_vertex_area?3:2,FLT_EPSILON,cache.dist.data(),cache.faces.data(),cache.vertices.data()); };
  calculate();
  if(step==3){
    bool retry=std::any_of(cache.vertices.begin(),cache.vertices.end(),[](const VertexField& field){return field.retry!=0;});
    if(retry){
      std::fprintf(stderr,"GEOM_DEGENERATE native-normal-retry\n");
      MRISMP_computeNormals(mesh,false);calculate();
    } else {
      if(audit) MRISMP_computeNormals(mesh,false);
      for(int id=0;id<mesh->nvertices;++id){auto& field=cache.vertices[id];
        float* components[]={mesh->v_nx,mesh->v_ny,mesh->v_nz};
        for(int axis=0;axis<3;++axis){if(audit) equal(components[axis][id],field.normal[axis],"vertex-normal",id);components[axis][id]=field.normal[axis];}
        if(audit) equal(mesh->v_area[id],field.initialArea,"corner-area",id);
        if(mesh->v_origarea[id]<0) mesh->v_origarea[id]=field.initialArea;
        mesh->v_area[id]=field.initialArea;
      }
    }
  }
  MRISMP_computeSurfaceDimensions(mesh);
  if(audit) MRISMP_computeVertexDistances(mesh);
  if(!mesh->v_dist_buffer) mesh->v_dist_buffer=(float**)calloc(mesh->nvertices,sizeof(float*));
  for(int id=0;id<mesh->nvertices;++id){
    MRISMP_makeDist(mesh,id); int count=cache.row[id+1]-cache.row[id];
    if(mesh->v_dist_capacity[id]<count) throw std::runtime_error("distance buffer too small");
    const float* values=cache.dist.data()+cache.row[id];
    if(audit) for(int slot=0;slot<count;++slot) equal(mesh->v_dist[id][slot],values[slot],"edge-distance",cache.row[id]+slot);
    std::memcpy(mesh->v_dist[id],values,count*sizeof(float));
  }
  mesh->dist_nsize=mesh->nsize;
  if(step<2 || audit) MRISMP_computeTriangleProperties(mesh);
  if(step>=2){
    for(int face=0;face<mesh->nfaces;++face){auto& field=cache.faces[face];
      if(audit){equal(mesh->f_area[face],field.area,"face-area",face);
        auto& normal=mesh->f_norm[face]; equal(normal.x,field.normal[0],"face-nx",face);equal(normal.y,field.normal[1],"face-ny",face);equal(normal.z,field.normal[2],"face-nz",face);
        for(int axis=0;axis<3;++axis) equal(mesh->f_angle[face][axis],field.angles[axis],"face-angle",face);
      }
      mesh->f_area[face]=field.area;mesh->f_normSet[face]=true;mesh->f_norm[face]={field.normal[0],field.normal[1],field.normal[2]};
      for(int corner=0;corner<3;++corner) mesh->f_angle[face][corner]=field.angles[corner];
    }
    for(int id=0;id<mesh->nvertices;++id){if(audit) equal(mesh->v_area[id],cache.vertices[id].area,"vertex-area",id);mesh->v_area[id]=cache.vertices[id].area;}
    float total=sumFaces(mesh);if(audit) equal(mesh->total_area,total,"total-area",0);mesh->total_area=total;
  }
  mesh->avg_vertex_area=mesh->total_area/mesh->nvertices;
  MRISMP_computeAvgInterVertexDist(mesh,&mesh->std_vertex_dist);mrismp_OrientSurface(mesh);
  if(mesh->status==MRIS_PARAMETERIZED_SPHERE || mesh->status==MRIS_RIGID_BODY || mesh->status==MRIS_SPHERE)
    mesh->total_area=M_PI*mesh->radius*mesh->radius*4.0;
  std::fprintf(stderr,"GEOM_PASS %d %d %d\n",step,int(audit),int(MRIS_Status_distanceFormula(mesh->status)));
}
