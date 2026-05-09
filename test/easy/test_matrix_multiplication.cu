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

void helper(const std::vector<float>& expected, const std::vector<float>& A,
            const std::vector<float>& B, std::vector<float>& C, int M, int N,
            int K) {
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, A.size() * sizeof(float));
  cudaMalloc(&d_B, B.size() * sizeof(float));
  cudaMalloc(&d_C, C.size() * sizeof(float));

  cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
  solve(d_A, d_B, d_C, M, N, K);
  cudaMemcpy(C.data(), d_C, C.size() * sizeof(float), cudaMemcpyDeviceToHost);

  EXPECT_EQ(expected, C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

// https://leetgpu.com/challenges/matrix-multiplication
TEST(LeetGPU, MatrixMultiplication) {
  std::vector<float> A1 = {1.0, 2.0, 3.0, 4.0};
  std::vector<float> B1 = {5.0, 6.0, 7.0, 8.0};
  std::vector<float> C1 = {0, 0, 0, 0};
  std::vector<float> expected1 = {19.0, 22.0, 43.0, 50.0};
  helper(expected1, A1, B1, C1, 2, 2, 2);

  std::vector<float> A2 = {1.0, 2.0, 3.0};
  std::vector<float> B2 = {4.0, 5.0, 6.0};
  std::vector<float> C2 = {0};
  std::vector<float> expected2 = {32.0};
  helper(expected2, A2, B2, C2, 1, 3, 1);
}