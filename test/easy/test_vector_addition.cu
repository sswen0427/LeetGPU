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

void helper(const std::vector<float>& expected, const std::vector<float>& A,
            const std::vector<float>& B, std::vector<float>& C) {
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, A.size() * sizeof(float));
  cudaMalloc(&d_B, B.size() * sizeof(float));
  cudaMalloc(&d_C, C.size() * sizeof(float));

  cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
  solve(d_A, d_B, d_C, A.size());
  cudaMemcpy(C.data(), d_C, C.size() * sizeof(float), cudaMemcpyDeviceToHost);

  EXPECT_EQ(expected, C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

// https://leetgpu.com/challenges/vector-addition
TEST(LeetGPU, VectorAddition) {
  std::vector<float> A1 = {1.0, 2.0, 3.0, 4.0};
  std::vector<float> B1 = {5.0, 6.0, 7.0, 8.0};
  std::vector<float> C1 = {0, 0, 0, 0};
  std::vector<float> expected1 = {6.0, 8.0, 10.0, 12.0};
  helper(A1, B1, C1, expected1);

  std::vector<float> A2 = {1.5, 1.5, 1.5};
  std::vector<float> B2 = {2.3, 2.3, 2.3};
  std::vector<float> C2 = {0, 0, 0};
  std::vector<float> expected2 = {3.8, 3.8, 3.8};
  helper(A2, B2, C2, expected2);
}