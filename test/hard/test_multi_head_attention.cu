#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

__global__ void scale(const float* Q, float* QD, int n, float factor) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    QD[id] = Q[id] * factor;
  }
}

__global__ void qk(const float* QD, const float* K, float* QK, int N,
                   int d_model) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < N * N) {
    int row = id / N;
    int col = id % N;

    int index1 = row * d_model;
    int index2 = col * d_model;
    float sum = 0;
    for (int i = 0; i < N; i++) {
      sum += QD[index1 + i] * K[index2 + i];
    }
    QK[id] = sum;
  }
}

__global__ void softmax(float* QK, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++) {
      max_val = fmaxf(max_val, QK[row * N + i]);
    }

    float sum = 0;
    for (int i = 0; i < N; i++) {
      float e_val = expf(QK[row * N + i] - max_val);
      QK[row * N + i] = e_val;
      sum += e_val;
    }
    for (int i = 0; i < N; i++) {
      QK[row * N + i] /= sum;
    }
  }
}

__global__ void mm(const float* QK, const float* V, float* output, int N,
                   int d_model) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < N * d_model) {
    float result = 0;
    int row = id / d_model;
    int column = id % d_model;
    for (int i = 0; i < N; i++) {
      result += QK[row * N + i] * V[i * d_model + column];
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
  cudaMalloc(&QK, N * N * sizeof(float));

  int threadsPerBlock = 1024;

  float factor = 1.0f / std::sqrt(1.0f * d_model / h);
  int blocksPerGrid = (N * d_model + threadsPerBlock - 1) / threadsPerBlock;
  scale<<<blocksPerGrid, threadsPerBlock>>>(Q, QD, N * d_model, factor);

  blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;
  qk<<<blocksPerGrid, threadsPerBlock>>>(QD, K, QK, N, d_model);
  softmax<<<blocksPerGrid, threadsPerBlock>>>(QK, N);

  blocksPerGrid = (N * d_model + threadsPerBlock - 1) / threadsPerBlock;
  mm<<<blocksPerGrid, threadsPerBlock>>>(QK, V, output, N, d_model);
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
  std::vector<float> k1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  std::vector<float> V1 = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
  std::vector<float> expected1 = {2.39, 2.89, 3.50, 4.00,
                                  2.50, 3.00, 3.50, 4.00};
  helper(expected1, Q1, K1, V1, N1, d_model1, h1);

  int N2 = 1, d_model2 = 2, h2 = 1;
  std::vector<float> Q2 = {1.0, 1.0};
  std::vector<float> k2 = {1.0, 1.0};
  std::vector<float> V2 = {2.0, 3.0};
  std::vector<float> expected2 = {2.0, 3.0};
  helper(expected2, Q2, K2, V2, N2, d_model2, h2);
}