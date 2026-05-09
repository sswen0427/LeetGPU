#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
  unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < N) {
    C[id] = A[id] + B[id];
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
  cudaDeviceSynchronize();
}

// https://leetgpu.com/challenges/vector-addition
TEST(LeetGPU, test1) {
  float A[] = {1.0, 2.0, 3.0, 4.0};
  float B[] = {5.0, 6.0, 7.0, 8.0};
  float C[] = {0, 0, 0, 0};

  float *d_A, *d_B, *d_C;
  constexpr size_t size = 4 * sizeof(float);
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
  solve(d_A, d_B, d_C, 4);
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 4; i++) {
    constexpr float expected[] = {6.0, 8.0, 10.0, 12.0};
    EXPECT_EQ(expected[i], C[i]);
  }
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
