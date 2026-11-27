#pragma once

struct FaceField { float normal[3], area, angles[3]; };
struct VertexField { float normal[3], initialArea, area; int retry; };

extern "C" void geomInit(int vertices,int faces,const int* faceIds,const int* offsets,const int* neighbors,
                         const int* faceOffsets,const int* corners,const float* angleTable);
extern "C" void geomCompute(const float* coordX,const float* coordY,const float* coordZ,int step,int formula,
                            int unitize,int areaDivisor,float epsilon,float* distances,FaceField* faces,VertexField* vertices);
