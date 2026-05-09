#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B,
                                             float* C, int M, int N, int K) {
  int column = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < M && column < K) {
    float result = 0;
    for (int i = 0; i < N; i++) {
      result += A[row * N + i] * B[i * K + column];
    }
    C[row * K + column] = result;
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                   N, K);
  cudaDeviceSynchronize();
}

// https://leetgpu.com/challenges/matrix-multiplication
TEST(LeetGPU, MatrixMultiplication) {
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
  solve(d_A, d_B, d_C, 2, 2, 2);
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 4; i++) {
    constexpr float expected[] = {6.0, 8.0, 10.0, 12.0};
    EXPECT_EQ(expected[i], C[i]);
  }
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}