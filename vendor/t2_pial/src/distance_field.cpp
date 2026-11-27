#include <fstream>
#include <omp.h>
#include "mrisurf.h"
#include "lean_distance.h"

int main(int argc,char** argv){
  if(argc!=4 && argc!=5) return 2;
  std::string mode=argc==5?argv[4]:"native";
  if(mode!="native" && mode!="lean") return 2;
  omp_set_num_threads(1);
  MRI* mask=MRIread(argv[1]);
  if(!mask || mask->nframes!=1) return 3;
  MRI* field=MRIcloneDifferentType(mask,MRI_FLOAT);
  int label=std::stoi(argv[3]);
  auto result=mode=="lean"?leanSignedField(mask,field,label):MRIdistanceTransform(mask,field,label,100,DTRANS_MODE_SIGNED,nullptr);
  if(result!=field || !field->ischunked || !field->chunk) return 4;
  std::ofstream output(argv[2],std::ios::binary);
  output.write(reinterpret_cast<const char*>(&field->outside_val),sizeof(field->outside_val));
  output.write(static_cast<const char*>(field->chunk),field->bytes_total); output.close();
  MRIfree(&field); MRIfree(&mask);
  return output ? 0 : 5;
}
