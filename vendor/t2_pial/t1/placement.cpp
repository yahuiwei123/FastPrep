#include "mrisurf.h"
#include "mrishash.h"
#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

extern double orderedMove(MRIS*,float,float,MHT*,float);
extern "C" void smoothSigned(float*,const int*,const int*,const unsigned char*,int,int);

void CMDargNErr(char* option,int count) { std::fprintf(stderr,"%s requires %d arguments\n",option,count); std::exit(2); }
int CMDsingleDash(char* flag) { return flag[0] == '-' && flag[1] != '-'; }
int outputHelpXml(const unsigned char* text,unsigned int size) { return std::fwrite(text,1,size,stdout) == size ? 0 : 1; }

namespace {
const std::string mode = std::getenv("T1_MODE") ? std::getenv("T1_MODE") : "native";
const bool trace = std::getenv("T1_TRACE") && std::string(std::getenv("T1_TRACE")) == "1";
int roundIndex = 0, attempts = 0, averages = 0, restores = 0;
using Clock = std::chrono::steady_clock;
template<class Function> Function original(const char* symbol) {
  auto address = dlsym(RTLD_NEXT, symbol);
  if (!address) throw std::runtime_error(symbol);
  return reinterpret_cast<Function>(address);
}
void timing(const char* name, Clock::time_point start) {
  static std::ofstream stream("timing.tsv");
  stream << roundIndex << '\t' << name << '\t' << std::setprecision(12)
         << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
}
void capture(MRIS* surface, const char* name, int index, int fields) {
  if (!trace) return;
  std::vector<float> values;
  values.reserve(size_t(surface->nvertices) * fields);
  for (int vertex = 0; vertex < surface->nvertices; ++vertex) {
    const auto& point = surface->vertices[vertex];
    if (fields == 6) values.insert(values.end(), {point.dx,point.dy,point.dz,point.tdx,point.tdy,point.tdz});
    else if (fields == 9) values.insert(values.end(), {point.targx,point.targy,point.targz,point.val,point.d,
                                                        point.val2,point.mean,float(point.marked),float(point.ripflag)});
    else values.insert(values.end(), {point.x,point.y,point.z});
  }
  std::string path = std::string(name) + "-" + std::to_string(roundIndex) + "-" + std::to_string(index) + ".f32";
  std::ofstream stream(path, std::ios::binary);
  if (!stream.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(float))) throw std::runtime_error(path);
}
}

int MRISpositionSurface(MRIS* surface,MRI* brain,MRI* smooth,INTEGRATION_PARMS* params) {
  static auto native = original<decltype(&MRISpositionSurface)>("_Z19MRISpositionSurfaceP4MRISP3MRIS2_P17INTEGRATION_PARMS");
  capture(surface,"target",0,9);
  attempts = averages = restores = 0;
  auto start = Clock::now();
  int status = native(surface,brain,smooth,params);
  timing("position-inclusive",start); capture(surface,"position",0,3); ++roundIndex;
  return status;
}

int mrisAverageSignedGradients(MRIS* surface,int repeats) {
  static auto native = original<decltype(&mrisAverageSignedGradients)>("_Z26mrisAverageSignedGradientsP4MRISi");
  auto start = Clock::now();
  int status = 0;
  if (mode != "smooth" && mode != "combined") status = native(surface,repeats);
  else if (repeats > 0) {
    int count = surface->nvertices;
    std::vector<float> gradients(3*count);
    std::vector<unsigned char> frozen(count);
    std::vector<int> offsets(count+1), neighbors;
    neighbors.reserve(6*count);
    for (int vertex = 0; vertex < count; ++vertex) {
      const auto& point = surface->vertices[vertex];
      const auto& topology = surface->vertices_topology[vertex];
      gradients[3*vertex] = point.dx; gradients[3*vertex+1] = point.dy; gradients[3*vertex+2] = point.dz;
      frozen[vertex] = point.ripflag != 0;
      if (topology.vnum) neighbors.insert(neighbors.end(), topology.v, topology.v + topology.vnum);
      offsets[vertex+1] = neighbors.size();
    }
    smoothSigned(gradients.data(),offsets.data(),neighbors.data(),frozen.data(),count,repeats);
    for (int vertex = 0; vertex < count; ++vertex) if (!frozen[vertex]) {
      auto& point = surface->vertices[vertex];
      point.dx = point.tdx = gradients[3*vertex]; point.dy = point.tdy = gradients[3*vertex+1];
      point.dz = point.tdz = gradients[3*vertex+2];
    }
  }
  timing("signed-average",start); capture(surface,"gradient",averages++,6); return status;
}

double mrisAsynchronousTimeStep(MRIS* surface,float momentum,float step,MHT* table,float limit) {
  static auto native = original<decltype(&mrisAsynchronousTimeStep)>("_Z24mrisAsynchronousTimeStepP4MRISffP15MRIS_HASH_TABLEf");
  auto start = Clock::now();
  auto result = (mode == "ordered" || mode == "combined") ? orderedMove(surface,momentum,step,table,limit)
                                                          : native(surface,momentum,step,table,limit);
  timing("collision-aware-move",start); capture(surface,"move",attempts++,3); return result;
}

int MRISrestoreVertexPositions(MRIS* surface,int which) {
  static auto native = original<decltype(&MRISrestoreVertexPositions)>("_Z26MRISrestoreVertexPositionsP4MRISi");
  int status = native(surface,which);
  if (which == TMP2_VERTICES) capture(surface,"restore",restores++,3);
  return status;
}

int MRIScomputeBorderValues(MRIS* surface,MRI* volume,MRI* smooth,double inside,double high,double low,double outsideLow,
                            double outsideHigh,double sigma,float distance,FILE* log,int kind,MRI* mask,double threshold,
                            int flags,MRI* seg,int first,int last,int verbose) {
  static auto native = original<decltype(&MRIScomputeBorderValues)>("_Z23MRIScomputeBorderValuesP4MRISP3MRIS2_ddddddfP8_IO_FILEiS2_diS2_iii");
  auto start = Clock::now();
  int status = native(surface,volume,smooth,inside,high,low,outsideLow,outsideHigh,sigma,distance,log,kind,mask,threshold,flags,seg,first,last,verbose);
  timing("border-values",start); return status;
}

int mrisComputeIntensityTerm(MRIS* surface,double weight,MRI* brain,MRI* smooth,double sigma,INTEGRATION_PARMS* params) {
  static auto native = original<decltype(&mrisComputeIntensityTerm)>("_Z24mrisComputeIntensityTermP4MRISdP3MRIS2_dP17INTEGRATION_PARMS");
  auto start = Clock::now(); int status = native(surface,weight,brain,smooth,sigma,params);
  timing("intensity-force",start); return status;
}

int mrisComputeRepulsiveTerm(MRIS* surface,double weight,MHT* vertices,MHT* faces) {
  static auto native = original<decltype(&mrisComputeRepulsiveTerm)>("_Z24mrisComputeRepulsiveTermP4MRISdP15MRIS_HASH_TABLES2_");
  auto start = Clock::now(); int status = native(surface,weight,vertices,faces);
  timing("self-repulsion",start); return status;
}

int mrisComputeSurfaceRepulsionTerm(MRIS* surface,double weight,MHT* vertices) {
  static auto native = original<decltype(&mrisComputeSurfaceRepulsionTerm)>("_Z31mrisComputeSurfaceRepulsionTermP4MRISdP15MRIS_HASH_TABLE");
  auto start = Clock::now(); int status = native(surface,weight,vertices);
  timing("reference-repulsion",start); return status;
}
