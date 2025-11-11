#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_33_BatchNorm_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

// 定义 warp 大小常量
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// [重要] CUDA 辅助函数 (在 kernel 之前声明/定义)
__device__ float blockReduceSum(float val, float* shared) {
    // Warp-level reduce using shuffle
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    // Reduce within warp
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // Write warp result to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Final reduce within first warp
    int warpCount = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < warpCount) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// 计算每个通道的 sum 和 sumsq 的 kernel（跨 N,H,W 归约）
__global__ void reduce_sums_kernel(
    const float* __restrict__ x,
    float* __restrict__ sum,
    float* __restrict__ sumsq,
    int C,
    long long M,              // M = N*H*W
    long long strideCW,       // H*W
    long long strideNC        // C*H*W
) {
    extern __shared__ float shmem[]; // 大小为 (blockDim.x / WARP_SIZE)
    int c = blockIdx.x;              // 当前通道
    if (c >= C) return;

    // 全局跨线程步长（覆盖 M）
    long long globalStride = (long long)blockDim.x * (long long)gridDim.y;
    long long start = (long long)threadIdx.x + (long long)blockIdx.y * (long long)blockDim.x;

    float local_sum   = 0.0f;
    float local_sumsq = 0.0f;

    for (long long i = start; i < M; i += globalStride) {
        // 索引计算：idx = n*strideNC + c*strideCW + hw
        long long n  = i / strideCW;
        long long hw = i % strideCW;
        long long idx = n * strideNC + (long long)c * strideCW + hw;
        float v = __ldg(x + idx);
        local_sum   += v;
        local_sumsq += v * v;
    }

    // 块内归约到一个值
    float block_sum   = blockReduceSum(local_sum, shmem);
    // 只有线程0将结果原子加到全局
    if (threadIdx.x == 0) {
        atomicAdd(&sum[c], block_sum);
    }

    float block_sumsq = blockReduceSum(local_sumsq, shmem);
    if (threadIdx.x == 0) {
        atomicAdd(&sumsq[c], block_sumsq);
    }
}

// 根据 sum/sumsq 计算 mean 和 invstd 的 kernel
__global__ void compute_stats_kernel(
    const float* __restrict__ sum,
    const float* __restrict__ sumsq,
    float* __restrict__ mean,
    float* __restrict__ invstd,
    long long M,    // N*H*W
    float eps,
    int C
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float m = sum[c] / (float)M;
    float v = sumsq[c] / (float)M - m * m;
    v = fmaxf(v, 0.0f); // 数值稳定
    mean[c]   = m;
    invstd[c] = rsqrtf(v + eps);
}

// 归一化输出的 kernel：y = (x - mean[c]) * invstd[c]
__global__ void normalize_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    int C,
    long long M,          // N*H*W
    long long strideCW,   // H*W
    long long strideNC    // C*H*W
) {
    int c = blockIdx.x; // 当前通道
    if (c >= C) return;

    float m   = mean[c];
    float inv = invstd[c];

    long long globalStride = (long long)blockDim.x * (long long)gridDim.y;
    long long start = (long long)threadIdx.x + (long long)blockIdx.y * (long long)blockDim.x;

    for (long long i = start; i < M; i += globalStride) {
        long long n  = i / strideCW;
        long long hw = i % strideCW;
        long long idx = n * strideNC + (long long)c * strideCW + hw;
        float xv = __ldg(x + idx);
        y[idx] = (xv - m) * inv;
    }
}

// C++ Wrapper 实现
torch::Tensor kb_1_33_BatchNorm_wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "arg0 must be a CUDA tensor");
    TORCH_CHECK(arg0.dtype() == torch::kFloat32, "arg0 must be float32");
    TORCH_CHECK(arg0.dim() == 4, "arg0 must be 4D tensor [N, C, H, W]");

    // 确保连续
    auto x = arg0.contiguous();

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    TORCH_CHECK(C > 0 && N > 0 && H > 0 && W > 0, "Invalid tensor dimensions");

    auto y = torch::empty_like(x);

    // 分配辅助张量（在 GPU 上）
    auto options_f = x.options().dtype(torch::kFloat32);
    auto sum    = torch::zeros({C}, options_f);
    auto sumsq  = torch::zeros({C}, options_f);
    auto mean   = torch::empty({C}, options_f);
    auto invstd = torch::empty({C}, options_f);

    const long long M = (long long)N * (long long)H * (long long)W;
    const long long strideCW = (long long)H * (long long)W;
    const long long strideNC = (long long)C * strideCW;

    // 配置 kernel 参数
    int threads = 512; // Increased to 512 threads for 16 warps/block to boost occupancy without adding blocks
    // 让每个线程大约处理 ~32 个元素（可根据需要调整）
    long long targetElemsPerThread = 32;
    long long totalThreadsPerChannel = (long long)threads;
    long long blocksY = (M + totalThreadsPerChannel * targetElemsPerThread - 1) /
                        (totalThreadsPerChannel * targetElemsPerThread);
    if (blocksY < 1) blocksY = 1;
    if (blocksY > 65535) blocksY = 65535; // 避免超过维度限制

    dim3 grid_reduce((unsigned int)C, (unsigned int)blocksY, 1);
    dim3 block_reduce(threads, 1, 1);
    size_t shmem_size = (threads / WARP_SIZE) * sizeof(float); // 用于 blockReduceSum

    auto stream = at::cuda::getCurrentCUDAStream();

    // 启动归约 kernel，计算每个通道的 sum 和 sumsq
    reduce_sums_kernel<<<grid_reduce, block_reduce, shmem_size, stream>>>(
        x.data_ptr<float>(),
        sum.data_ptr<float>(),
        sumsq.data_ptr<float>(),
        (int)C,
        M,
        strideCW,
        strideNC
    );

    // 计算 mean 和 invstd
    int statsThreads = 256;
    int statsBlocks = (int)((C + statsThreads - 1) / statsThreads);
    compute_stats_kernel<<<statsBlocks, statsThreads, 0, stream>>>(
        sum.data_ptr<float>(),
        sumsq.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        M,
        1e-5f,  // epsilon
        (int)C
    );

    // 归一化输出
    dim3 grid_norm((unsigned int)C, (unsigned int)blocksY, 1);
    dim3 block_norm(threads, 1, 1);
    normalize_kernel<<<grid_norm, block_norm, 0, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        (int)C,
        M,
        strideCW,
        strideNC
    );

    // 返回输出张量
    return y;
}