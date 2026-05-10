#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <vector>

__global__ void scale(const float* Q, float* QD, int n, float factor) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    QD[id] = Q[id] * factor;
  }
}

__global__ void qk(const float* QD, const float* K, float* QK, int N,
                   int d_model, int h) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int d_k = d_model / h;
  if (id < h * N * N) {
    int row = id / (h * N);
    int col = id % (h * N);
    int head_idx = col / d_k;
    int head_col = col % d_k;

    int q_idx = row * d_model + head_idx * d_k;
    int k_idx = head_col * d_model + head_idx * d_k;

    float sum = 0;
    for (int i = 0; i < d_k; i++) {
      sum += QD[q_idx + i] * K[k_idx + i];
    }
    QK[id] = sum;
  }
}

__global__ void softmax(float* QK, int N, int h) {
  int row_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_id < h * N) {
    int index = row_id * N;

    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++) {
      max_val = fmaxf(max_val, QK[index + i]);
    }

    float sum = 0;
    for (int i = 0; i < N; i++) {
      float e_val = expf(QK[index + i] - max_val);
      QK[index + i] = e_val;
      sum += e_val;
    }
    for (int i = 0; i < N; i++) {
      QK[index + i] /= sum;
    }
  }
}

__global__ void mm(const float* QK, const float* V, float* output, int N,
                   int d_model, int h) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int d_k = d_model / h;
  if (id < N * d_model) {
    int row = id / d_model;
    int col = id % d_model;
    int head_idx = col / d_k;

    float result = 0;
    for (int i = 0; i < N; i++) {
      int index1 = row * d_model + head_idx * d_k + i;
      int index2 = i * d_model + col;
      result += QK[index1] * V[index2];
    }
    output[id] = result;
  }
}

// Q, K, V, output are device pointers
void solve(const float* Q, const float* K, const float* V, float* output, int N,
           int d_model, int h) {
  float* QD;
  float* QK;
  cudaMalloc(&QD, N * d_model * sizeof(float));
  cudaMalloc(&QK, h * N * N * sizeof(float));

  int threadsPerBlock = 1024;

  float factor = 1.0f / std::sqrt(1.0f * d_model / h);
  int blocksPerGrid = (N * d_model + threadsPerBlock - 1) / threadsPerBlock;
  scale<<<blocksPerGrid, threadsPerBlock>>>(Q, QD, N * d_model, factor);

  blocksPerGrid = (h * N * N + threadsPerBlock - 1) / threadsPerBlock;
  qk<<<blocksPerGrid, threadsPerBlock>>>(QD, K, QK, N, d_model, h);
  std::vector<float> tmp(h * N * N);
  cudaMemcpy(tmp.data(), QK, h * N * N * sizeof(float), cudaMemcpyDeviceToHost);

  blocksPerGrid = (h * N + threadsPerBlock - 1) / threadsPerBlock;
  softmax<<<blocksPerGrid, threadsPerBlock>>>(QK, N, h);

  blocksPerGrid = (N * d_model + threadsPerBlock - 1) / threadsPerBlock;
  mm<<<blocksPerGrid, threadsPerBlock>>>(QK, V, output, N, d_model, h);

  cudaFree(QD);
  cudaFree(QK);
}

void helper(const std::vector<float>& expected, const std::vector<float>& Q,
            const std::vector<float>& K, const std::vector<float>& V, int N,
            int d_model, int h) {
  float *d_Q, *d_K, *d_V, *d_output;
  cudaMalloc(&d_Q, Q.size() * sizeof(float));
  cudaMalloc(&d_K, K.size() * sizeof(float));
  cudaMalloc(&d_V, V.size() * sizeof(float));
  cudaMalloc(&d_output, expected.size() * sizeof(float));

  cudaMemcpy(d_Q, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice);
  solve(d_Q, d_K, d_V, d_output, N, d_model, h);
  cudaDeviceSynchronize();

  std::vector<float> output(expected.size());
  cudaMemcpy(output.data(), d_output, output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  EXPECT_EQ(expected, output);
  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_output);
}

// https://leetgpu.com/challenges/multi-head-attention
TEST(LeetGPU, MultiHeadAttention) {
  int N1 = 2, d_model1 = 4, h1 = 2;
  std::vector<float> Q1 = {1.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  std::vector<float> K1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  std::vector<float> V1 = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
  std::vector<float> expected1 = {2.39, 2.89, 3.50, 4.00,
                                  2.50, 3.00, 3.50, 4.00};
  helper(expected1, Q1, K1, V1, N1, d_model1, h1);

  int N2 = 1, d_model2 = 2, h2 = 1;
  std::vector<float> Q2 = {1.0, 1.0};
  std::vector<float> K2 = {1.0, 1.0};
  std::vector<float> V2 = {2.0, 3.0};
  std::vector<float> expected2 = {2.0, 3.0};
  helper(expected2, Q2, K2, V2, N2, d_model2, h2);
}