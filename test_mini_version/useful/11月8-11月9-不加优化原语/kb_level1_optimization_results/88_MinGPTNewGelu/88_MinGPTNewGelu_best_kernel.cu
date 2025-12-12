#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_88_MinGPTNewGelu_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

// ======================================
// CUDA 常量
// ======================================
constexpr float kSqrt2OverPi = 0.7978845608028654f;   // √(2/π)
constexpr float kCoeff       = 0.044715f;             // 0.044715

// ======================================
// (可选) 归约辅助函数示例 —— 当前内核未使用
// 但给出一个 blockReduceSum 的参考实现，以备复用
// ======================================
__device__ float blockReduceSum(float val, float* shared)
{
    int lane = threadIdx.x & 31;          // 线程在 warp 内的索引
    int warp = threadIdx.x >> 5;          // warp 索引

    // Warp 内归约
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // 每个 warp 的 lane0 把结果写入 shared
    if (lane == 0)
        shared[warp] = val;

    __syncthreads();

    // blockDim.x / 32 个 warp 参与最终归约
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.f;

    // 第 0 个 warp 完成最后一次归约
    if (warp == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ======================================
// GELU 核心计算的 __device__ 内联函数
// ======================================
__device__ __forceinline__ float gelu(float x)
{
    float x_cube = x * x * x;
    float inner  = kSqrt2OverPi * (x + kCoeff * x_cube);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ======================================
// CUDA Kernel
// ======================================
__global__ void gelu_kernel(
        const float* __restrict__ input,
        float*       __restrict__ output,
        int64_t                numel)
{
    int64_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;

    while (idx < numel) {
        output[idx] = gelu(input[idx]);
        idx += stride;
    }
}

// ======================================
// C++ 包装函数
// ======================================
torch::Tensor kb_88_MinGPTNewGelu_wrapper(torch::Tensor arg0)
{
    TORCH_CHECK(arg0.is_cuda(), "Input must reside on CUDA device");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat,
                "Only float32 type is supported");
    TORCH_CHECK(arg0.is_contiguous(),
                "Input tensor must be contiguous");

    auto out = at::empty_like(arg0);

    const int64_t numel = arg0.numel();
    const int     threads = 256;
    const int     blocks  = static_cast<int>((numel + threads - 1) / threads);

    // 为了避免启动过多 block，限制上限 (硬件相关，可按需调整)
    const int     maxBlocks = 1024;
    const int     grid      = std::min(blocks, maxBlocks);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    gelu_kernel<<<grid, threads, 0, stream>>>(
        arg0.data_ptr<float>(),
        out.data_ptr<float>(),
        numel);

    // CUDA 内核调用后检查错误（可选）
    #ifdef __CUDA_ARCH__
    #else
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
    #endif

    return out;
}