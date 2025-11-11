#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_30_Softsign_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
// PyTorch 2.1+ 移除了 c10::cuda::getCurrentCUDAStream
// 使用 at::cuda::getCurrentCUDAStream() 代替
#include <ATen/cuda/CUDAContext.h>

// -----------------------------------------------------------------------------
// Fused elementwise pipeline configuration (compile-time selection)
// Users can define these macros at compile-time to enable fusion without
// changing the wrapper signature or incurring runtime branching overhead.
// Defaults correspond to softsign-only behavior.
// -----------------------------------------------------------------------------
#ifndef FUSE_ADD_CONST
#define FUSE_ADD_CONST 0
#endif

#ifndef FUSE_MUL_CONST
#define FUSE_MUL_CONST 0
#endif

#ifndef FUSE_TANH
#define FUSE_TANH 0
#endif

#ifndef FUSE_SIGMOID
#define FUSE_SIGMOID 0
#endif

#ifndef FUSE_RELU
#define FUSE_RELU 0
#endif

#ifndef ADD_CONST_C
#define ADD_CONST_C 0.0f
#endif

#ifndef MUL_CONST_C
#define MUL_CONST_C 1.0f
#endif

// [重要] 在此放置所有 CUDA 辅助函数 (例如 blockReduceSum)
// (确保它们在使用它们的 kernel 之前被定义)
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Warp 内归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 每个 warp 的第一个线程将结果写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 第一个 warp 进行最终归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// -----------------------------------------------------------------------------
// Operation helpers: composed to form a fused pipeline
// -----------------------------------------------------------------------------
__forceinline__ __device__ float op_softsign(float v) {
    // v / (1 + |v|), using fast divide
    return __fdividef(v, 1.0f + fabsf(v));
}

__forceinline__ __device__ float op_add_const(float v) {
#if FUSE_ADD_CONST
    v = v + ADD_CONST_C;
#endif
    return v;
}

__forceinline__ __device__ float op_mul_const(float v) {
#if FUSE_MUL_CONST
    v = v * MUL_CONST_C;
#endif
    return v;
}

__forceinline__ __device__ float op_tanh(float v) {
#if FUSE_TANH
    v = tanhf(v);
#endif
    return v;
}

__forceinline__ __device__ float op_sigmoid(float v) {
#if FUSE_SIGMOID
    // Fast sigmoid approximation using __expf for single-precision
    v = 1.0f / (1.0f + __expf(-v));
#endif
    return v;
}

__forceinline__ __device__ float op_relu(float v) {
#if FUSE_RELU
    v = v > 0.0f ? v : 0.0f;
#endif
    return v;
}

// Compose the pipeline: softsign -> add_const -> mul_const -> tanh -> sigmoid -> relu
// The latter ops are enabled/disabled by compile-time flags to avoid runtime branching.
__forceinline__ __device__ float apply_pipeline(float v) {
    v = op_softsign(v);
    v = op_add_const(v);
    v = op_mul_const(v);
    v = op_tanh(v);
    v = op_sigmoid(v);
    v = op_relu(v);
    return v;
}

// CUDA 内核实现: 可融合的 y = softsign(x) 后接最多 4 个逐元素操作
// 向量化处理：使用 float4 进行加载/存储，并采用 warp 级连续块分配实现完美合并访问
// 同时提供对齐检测与小尺寸回退路径
__global__ void softsign_forward_kernel(const float* __restrict__ x,
                                        float* __restrict__ y,
                                        int64_t n) {
    if (n <= 0) return;

    // 小尺寸快速路径（直接标量 grid-stride），避免额外开销
    // 对于非常小的 n，标量路径通常更高效
    const uintptr_t x_addr = reinterpret_cast<uintptr_t>(x);
    const uintptr_t y_addr = reinterpret_cast<uintptr_t>(y);
    const bool aligned16 = ((x_addr | y_addr) & 0xF) == 0; // 16B 对齐检测

    // 对于未对齐或较小规模的情况，使用标量路径
    if (!aligned16 || n < 1024) {
        int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

        for (int64_t i = idx; i < n; i += stride) {
            float v = x[i];
            y[i] = apply_pipeline(v);
        }
        return;
    }

    // 向量化路径（float4）
    const int64_t vec_n = n / 4;   // 完整的 float4 数量
    const int64_t tail  = n % 4;   // 尾部标量元素数量

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y);

    // Warp 级连续块调度
    const int lane = threadIdx.x & (warpSize - 1);
    const int warps_per_block = blockDim.x >> 5; // blockDim.x / 32
    const int warp_in_block = threadIdx.x >> 5;
    const int64_t global_warp_id = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
    const int64_t total_warps = static_cast<int64_t>(gridDim.x) * warps_per_block;

    // 每个线程在一个 warp-iteration 中处理的 float4 数量（结构化 ILP）
    constexpr int THR_ILP = 4; // 保持适度的 ILP 以兼顾寄存器和吞吐
    const int64_t elems_per_warp = static_cast<int64_t>(warpSize) * THR_ILP; // 以 float4 为单位

    // warp-stride 循环：每个 warp 处理连续的 elems_per_warp 个 float4 元素块
    for (int64_t warp_base = global_warp_id * elems_per_warp;
         warp_base < vec_n;
         warp_base += total_warps * elems_per_warp) {

        #pragma unroll
        for (int i = 0; i < THR_ILP; ++i) {
            int64_t idx_vec = warp_base + lane + static_cast<int64_t>(i) * warpSize;
            if (idx_vec >= vec_n) break;

            float4 v = x4[idx_vec];

            // 逐元素应用融合流水线：softsign 后接可选 fused ops
            float4 r;
            r.x = apply_pipeline(v.x);
            r.y = apply_pipeline(v.y);
            r.z = apply_pipeline(v.z);
            r.w = apply_pipeline(v.w);

            y4[idx_vec] = r;
        }
    }

    // 处理尾部（不足 4 个的剩余元素），仅由一个线程完成以避免竞争
    if (tail && blockIdx.x == 0 && threadIdx.x == 0) {
        int64_t base = vec_n * 4;
        for (int64_t t = 0; t < tail; ++t) {
            float v = x[base + t];
            y[base + t] = apply_pipeline(v);
        }
    }
}

// C++ Wrapper 实现
torch::Tensor kb_30_Softsign_wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "kb_30_Softsign_wrapper: input must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat,
                "kb_30_Softsign_wrapper: only float32 tensors are supported");

    auto x = arg0.contiguous();
    auto out = at::empty_like(x);
    int64_t n = x.numel();

    if (n == 0) {
        return out;
    }

    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = out.data_ptr<float>();

    // 选择启动参数: 使用网格步长循环并限制网格尺寸以兼容所有设备
    constexpr int threads = 256;
    int64_t max_blocks = 65535; // 兼容性良好的上限
    int64_t needed_blocks = (n + threads - 1) / threads;
    int blocks = static_cast<int>(std::min<int64_t>(needed_blocks, max_blocks));

    auto stream = at::cuda::getCurrentCUDAStream();
    softsign_forward_kernel<<<blocks, threads, 0, stream.stream()>>>(x_ptr, y_ptr, n);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}