#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<class Value> struct Buffer {
  Value* data = nullptr;
  explicit Buffer(size_t count) { check(cudaMalloc(&data, count * sizeof(Value))); }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
};

__global__ void average(const float3* current, float3* next, const int* offsets, const int* neighbors,
                        const unsigned char* frozen, int count) {
  int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex >= count) return;
  float3 center = current[vertex];
  if (frozen[vertex]) { next[vertex] = center; return; }
  double sumX = center.x, sumY = center.y, sumZ = center.z;
  int accepted = 1;
  for (int slot = offsets[vertex]; slot < offsets[vertex + 1]; ++slot) {
    int adjacent = neighbors[slot];
    if (frozen[adjacent]) continue;
    float3 value = current[adjacent];
    if (value.x * center.x + value.y * center.y + value.z * center.z < 0) continue;
    sumX += value.x; sumY += value.y; sumZ += value.z; ++accepted;
  }
  next[vertex] = make_float3(sumX / accepted, sumY / accepted, sumZ / accepted);
}
}

extern "C" void smoothSigned(float* gradients, const int* offsets, const int* neighbors,
                             const unsigned char* frozen, int count, int repeats) {
  if (count <= 0 || repeats <= 0) return;
  Buffer<float3> first(count), second(count);
  Buffer<int> starts(count + 1), adjacent(offsets[count] ? offsets[count] : 1);
  Buffer<unsigned char> flags(count);
  check(cudaMemcpy(first.data, gradients, count * sizeof(float3), cudaMemcpyHostToDevice));
  check(cudaMemcpy(starts.data, offsets, (count + 1) * sizeof(int), cudaMemcpyHostToDevice));
  if (offsets[count]) check(cudaMemcpy(adjacent.data, neighbors, offsets[count] * sizeof(int), cudaMemcpyHostToDevice));
  check(cudaMemcpy(flags.data, frozen, count, cudaMemcpyHostToDevice));
  for (int iteration = 0; iteration < repeats; ++iteration) {
    average<<<(count + 255) / 256, 256>>>(first.data, second.data, starts.data, adjacent.data, flags.data, count);
    check(cudaGetLastError());
    std::swap(first.data, second.data);
  }
  check(cudaMemcpy(gradients, first.data, count * sizeof(float3), cudaMemcpyDeviceToHost));
}
