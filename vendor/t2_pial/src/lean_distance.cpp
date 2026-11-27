#include "lean_distance.h"
#include "lean_distance_generated.h"

bool compactHeapBuild(){
#ifdef T2GPU_COMPACT_HEAP
  return true;
#else
  return false;
#endif
}

MRI* leanSignedField(MRI* mask,MRI* field,int label){
  if(compactHeapBuild() && (!field || !field->ischunked || !field->chunk || field->type!=MRI_FLOAT || field->nframes!=1 || field->vox_total>INT_MAX))
    return MRIdistanceTransform(mask,field,label,100,DTRANS_MODE_SIGNED,nullptr);
  return t2lean::transform(mask,field,label,100,DTRANS_MODE_SIGNED,nullptr);
}
