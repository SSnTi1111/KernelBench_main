#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_29_Softplus_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
// PyTorch 2.1+ 移除了 c10::cuda::getCurrentCUDAStream
// 使用 at::cuda::getCurrentCUDAStream() 代替
#include <ATen/cuda/CUDAContext.h>

// [重要] 在此放置所有 CUDA 辅助函数 (例如 blockReduceSum)
// (确保它们在使用它们的 kernel 之前被定义)
__device__ float blockReduceSum(float val, float* shared) {
    // 示例 Warp 内归约（此示例在本内核中未使用，保留以供扩展）
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Warp 内归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 每个 warp 的第一个线程写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 第一个 warp 做最终归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

__device__ __forceinline__ float softplus_func(const float beta, float v) {
    float t = beta * v;
    float at = fabsf(t);
    float max_t = fmaxf(t, 0.0f);
    float exp_neg_at = expf(-at);
    float log_term = log1pf(exp_neg_at);
    return (max_t + log_term) / beta;
}

// CUDA 内核实现: Softplus with beta=1, threshold=20 (同 PyTorch 默认)
__global__ void softplus_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned long long N,
    const float beta,
    const float threshold
) {
    // 避免未使用参数的编译器警告（threshold 在分支消除后不再使用）
    (void)threshold;

    // 以 4 元素为一组的索引和步长，便于对齐的 float4 访问
    unsigned long long thread_linear = (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x + (unsigned long long)threadIdx.x;
    unsigned long long idx = thread_linear * 4ULL;
    unsigned long long stride = (unsigned long long)gridDim.x * (unsigned long long)blockDim.x * 4ULL;

    for (; idx < N; idx += stride) {
        unsigned long long remaining = N - idx;

        if (remaining >= 4ULL) {
            // 满 4 元素路径：向量化加载/存储
            const float* __restrict__ px = x + idx;
            float* __restrict__ py = y + idx;

            float4 xv = *reinterpret_cast<const float4*>(px);

            float4 outv;
            outv.x = softplus_func(beta, xv.x);
            outv.y = softplus_func(beta, xv.y);
            outv.z = softplus_func(beta, xv.z);
            outv.w = softplus_func(beta, xv.w);

            *reinterpret_cast<float4*>(py) = outv;
        } else if (remaining > 0ULL) {
            // 边界路径：逐标量处理剩余元素
            for (unsigned long long k = 0; k < remaining; ++k) {
                float v = x[idx + k];
                y[idx + k] = softplus_func(beta, v);
            }
        }
    }
}

// C++ Wrapper 实现
torch::Tensor kb_1_29_Softplus_wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "kb_1_29_Softplus_wrapper: input must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "kb_1_29_Softplus_wrapper: only float32 is supported");

    auto x = arg0.contiguous();
    auto out = torch::empty_like(x);

    const unsigned long long N = static_cast<unsigned long long>(x.numel());
    if (N == 0) {
        return out;
    }

    // 配置 kernel
    const int threads = 256;
    unsigned long long blocks_ull = (N + threads - 1ULL) / threads;
    int max_blocks = 65535; // 为兼容性使用较保守的最大 blocks 数
    int blocks = static_cast<int>(blocks_ull > static_cast<unsigned long long>(max_blocks) ? max_blocks : blocks_ull);

    const float beta = 1.0f;
    const float threshold = 20.0f;

    auto stream = at::cuda::getCurrentCUDAStream();
    softplus_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        beta,
        threshold
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "softplus_kernel launch failed: ", cudaGetErrorString(err));

    return out;
}