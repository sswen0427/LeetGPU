#include <cuda_runtime_api.h>
#include <float.h>
#include <gmock/gmock.h>
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
  EXPECT_THAT(output,
              ::testing::Pointwise(::testing::FloatNear(0.01f), expected));
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

  int N3 = 4, d_model3 = 4, h3 = 4;
  std::vector<float> Q3 = {
      0.693892240524292,   0.0009042024612426758, 0.9454827308654785,
      0.23009324073791504, 0.6322250366210938,    0.38155245780944824,
      0.27033376693725586, -0.6478875875473022,   0.8313214778900146,
      -0.5063142776489258, -0.5240383148193359,   -0.03675723075866699,
      -0.171423077583313,  -0.7483676075935364,   -0.6542049646377563,
      0.35372793674468994};
  std::vector<float> K3 = {
      -0.20068615674972534, -0.9310938119888306,  -0.3708909749984741,
      -0.9672888517379761,  -0.9191266298294067,  0.047138214111328125,
      -0.24324673414230347, -0.16725552082061768, -0.5620584487915039,
      -0.699795663356781,   -0.5322771072387695,  0.27606284618377686,
      0.8413174152374268,   -0.27321964502334595, -0.7728527188301086,
      0.6504384279251099};
  std::vector<float> V3 = {
      -0.09467095136642456, -0.09366124868392944, -0.12851965427398682,
      -0.45951956510543823, -0.9773003458976746,  -0.879672646522522,
      -0.6746125817298889,  -0.533287525177002,   -0.8082413673400879,
      0.7520341873168945,   -0.40661096572875977, -0.7008240222930908,
      -0.6610621809959412,  -0.8542488813400269,  0.7788692712783813,
      0.27696824073791504};
  std::vector<float> expected3 = {
      -0.6026404500007629,  -0.26905590295791626,  -0.1923728883266449,
      -0.3273615837097168,  -0.604112446308136,    -0.3411788046360016,
      -0.13302625715732574, -0.4150310754776001,   -0.6003673076629639,
      -0.17820903658866882, -0.056478358805179596, -0.35823899507522583,
      -0.648133397102356,   -0.13852502405643463,  -0.043359994888305664,
      -0.31218045949935913};
  helper(expected3, Q3, K3, V3, N3, d_model3, h3);
}