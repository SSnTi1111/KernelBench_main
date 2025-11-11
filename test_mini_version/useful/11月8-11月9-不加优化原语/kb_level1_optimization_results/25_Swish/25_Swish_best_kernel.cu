#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_25_Swish_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <cstdint>
// [!!! 关键 !!!] 
// PyTorch 2.1+ 移除了 c10::cuda::getCurrentCUDAStream
// 使用 at::cuda::getCurrentCUDAStream() 代替
#include <ATen/cuda/CUDAContext.h>

// [重要] 在此放置所有 CUDA 辅助函数 (例如 blockReduceSum)
// (确保它们在使用它们的 kernel 之前被定义)
__device__ float blockReduceSum(float val, float* shared) {
    // 简单的 Warp 归约 + 共享内存跨 Warp 归约
    int lane = threadIdx.x % warpSize;       // 线程在 warp 内的索引
    int wid  = threadIdx.x / warpSize;       // warp 的索引

    // Warp 内归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 每个 warp 的 lane 0 写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 第一个 warp 对各 warp 部分和再归约
    val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// 快速 Swish 计算（使用 fast-math exp 实现 sigmoid）
// Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float swish_fast(float v) {
    // 使用 __expf 获得更快的近似指数
    float s = 1.0f / (1.0f + __expf(-v));
    return v * s;
}

// [可选融合控制 - 常量内存开关]
// 通过在其他主机代码中使用 cudaMemcpyToSymbol 设置这些常量，
// 可以启用融合：y = scale * Swish(x + bias)
// - kb25_fuse_scale: 缩放因子（默认 1.0，表示不缩放）
// - kb25_bias_ptr: 偏置数组的设备指针（默认 nullptr，表示无偏置）
// - kb25_bias_len: 偏置长度（0 表示无偏置；1 表示标量广播；N 表示元素对齐；其他表示对长度为 bias_len 的循环广播）
__device__ __constant__ float kb25_fuse_scale = 1.0f;
__device__ __constant__ int   kb25_bias_len   = 0;
__device__ __constant__ const float* kb25_bias_ptr = nullptr;

// [新增] 数学近似与性能分析控制（默认保持关闭，保持向后兼容）
__device__ __constant__ int kb25_enable_approx = 0;      // 0: 关闭近似（使用 __expf），1: 启用近似
__device__ __constant__ int kb25_approx_mode   = 0;      // 近似模式：0=fast-exp(默认同原来)，1=fast-sigmoid，2=clipped-5th-poly
__device__ __constant__ int kb25_profile_enable = 0;     // 0: 关闭性能统计，1: 打开
__device__ __constant__ unsigned long long* kb25_profile_counters = nullptr; // [0]=launches, [1]=vec4_used, [2]=approx_used

// [新增] 近似 Sigmoid 实现
__device__ __forceinline__ float sigmoid_fast_sigmoid(float x) {
    // 有界且分段线性有理近似：0.5 * (x / (1 + |x|) + 1)
    float ax = fabsf(x);
    float t = x / (1.0f + ax);
    float s = 0.5f * (t + 1.0f);
    // s ∈ [0,1]
    return s;
}

__device__ __forceinline__ float sigmoid_poly5_clipped(float x) {
    // 五阶泰勒在 0 点的多项式：0.5 + x/4 - x^3/48 + x^5/480
    // 该多项式远离 0 会发散，故对结果夹紧到 [0,1]
    float x2 = x * x;
    float x3 = x2 * x;
    float x5 = x3 * x2;
    float s = 0.5f + (0.25f) * x - (1.0f / 48.0f) * x3 + (1.0f / 480.0f) * x5;
    // 夹紧
    s = fminf(fmaxf(s, 0.0f), 1.0f);
    return s;
}

// [新增] swish 选择（根据运行时常量选择近似/精确路径）
__device__ __forceinline__ float swish_select(float v, int approx_en, int approx_mode) {
    if (!approx_en) {
        // 和原始实现一致：使用 __expf 的快速近似
        float s = 1.0f / (1.0f + __expf(-v));
        return v * s;
    }
    // 选择近似模式
    float s;
    if (approx_mode == 1) {
        s = sigmoid_fast_sigmoid(v);
    } else if (approx_mode == 2) {
        // 使用五阶多项式并对结果夹紧
        s = sigmoid_poly5_clipped(v);
    } else {
        // 回退到 fast-exp
        s = 1.0f / (1.0f + __expf(-v));
    }
    return v * s;
}

// CUDA 内核实现（优化版本，支持向量化 float4 处理与标量回退）
// 保持签名不变以兼容现有 Wrapper
__global__ void kb_25_Swish_kernel(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   size_t N) {
    // 计算全局线程索引与步长
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    if (N == 0) return;

    // 读取融合控制常量到寄存器
    const float scale_c = kb25_fuse_scale;
    const int   bias_len_c = kb25_bias_len;
    const float* __restrict__ bias_ptr_c = kb25_bias_ptr;

    // 读取近似/性能控制到寄存器
    const int approx_en_c = kb25_enable_approx;
    const int approx_mode_c = kb25_approx_mode;
    const int profile_en_c = kb25_profile_enable;
    unsigned long long* prof_ptr = kb25_profile_counters;

    // 性能统计：仅在单线程做极低开销的计数
    if (profile_en_c && prof_ptr != nullptr) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            atomicAdd(&prof_ptr[0], 1ULL);              // kernel launches
            if (approx_en_c) atomicAdd(&prof_ptr[2], 1ULL); // approx used
        }
    }

    // 是否启用偏置
    const bool use_bias = (bias_ptr_c != nullptr) && (bias_len_c > 0);
    // 是否启用缩放（避免多余乘法）
    const bool use_scale = (scale_c != 1.0f);

    // 检查 16 字节对齐以决定是否进行 float4 向量化
    uintptr_t x_addr = reinterpret_cast<uintptr_t>(x);
    uintptr_t y_addr = reinterpret_cast<uintptr_t>(y);
    bool can_vec_xy = ((x_addr | y_addr) % 16u) == 0;

    // 对偏置对齐性进行检查（仅在需要偏置时）
    uintptr_t b_addr = reinterpret_cast<uintptr_t>(bias_ptr_c);
    bool bias_vec_aligned = use_bias && ((b_addr % 16u) == 0);

    // 向量化使用统计（仅一次）
    if (profile_en_c && prof_ptr != nullptr) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            if (can_vec_xy && N >= 4) {
                atomicAdd(&prof_ptr[1], 1ULL); // vec4 path used
            }
        }
    }

    // 如果无需融合（无偏置且 scale==1），执行原有高效路径
    if (!use_bias && !use_scale) {
        if (can_vec_xy && N >= 4) {
            // 对齐到4元素的向量化处理
            size_t N4 = N / 4;       // float4 个数
            size_t rem_base = N4 * 4;

            const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
            float4* __restrict__ y4 = reinterpret_cast<float4*>(y);

            // 向量化 grid-stride loop
            for (size_t i4 = tid; i4 < N4; i4 += stride) {
                float4 vx = x4[i4];

                // 逐通道计算 swish（根据选择的近似/精确）
                float4 vy;
                vy.x = swish_select(vx.x, approx_en_c, approx_mode_c);
                vy.y = swish_select(vx.y, approx_en_c, approx_mode_c);
                vy.z = swish_select(vx.z, approx_en_c, approx_mode_c);
                vy.w = swish_select(vx.w, approx_en_c, approx_mode_c);

                y4[i4] = vy;
            }

            // 处理尾部（不足4个的元素）
            for (size_t i = rem_base + tid; i < N; i += stride) {
                float v = x[i];
                y[i] = swish_select(v, approx_en_c, approx_mode_c);
            }
        } else {
            // 标量回退路径
            for (size_t i = tid; i < N; i += stride) {
                float v = x[i];
                y[i] = swish_select(v, approx_en_c, approx_mode_c);
            }
        }
        return;
    }

    // 以下为融合路径：y = scale * Swish(x + bias)
    // 针对常见情况提供向量化快速路径：
    // - bias_len == 1: 标量广播
    // - bias_len == N 且 bias 对齐: 与 x/y 同步向量化
    const bool bias_is_scalar = use_bias && (bias_len_c == 1);
    const bool bias_matches_N = use_bias && (static_cast<size_t>(bias_len_c) == N);

    if (can_vec_xy && N >= 4) {
        size_t N4 = N / 4;
        size_t rem_base = N4 * 4;

        const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
        float4* __restrict__ y4 = reinterpret_cast<float4*>(y);

        if (bias_is_scalar) {
            // 向量化 + 标量偏置广播
            const float b = bias_ptr_c[0];
            for (size_t i4 = tid; i4 < N4; i4 += stride) {
                float4 vx = x4[i4];

                // x + b
                vx.x += b; vx.y += b; vx.z += b; vx.w += b;

                // Swish
                float4 vy;
                vy.x = swish_select(vx.x, approx_en_c, approx_mode_c);
                vy.y = swish_select(vx.y, approx_en_c, approx_mode_c);
                vy.z = swish_select(vx.z, approx_en_c, approx_mode_c);
                vy.w = swish_select(vx.w, approx_en_c, approx_mode_c);

                // 可选缩放
                if (use_scale) {
                    vy.x *= scale_c; vy.y *= scale_c; vy.z *= scale_c; vy.w *= scale_c;
                }

                y4[i4] = vy;
            }

            // 尾部
            for (size_t i = rem_base + tid; i < N; i += stride) {
                float v = x[i] + b;
                float r = swish_select(v, approx_en_c, approx_mode_c);
                if (use_scale) r *= scale_c;
                y[i] = r;
            }
            return;
        }

        if (bias_matches_N && bias_vec_aligned) {
            // 向量化 + 等长偏置（对齐）
            const float4* __restrict__ b4 = reinterpret_cast<const float4*>(bias_ptr_c);
            for (size_t i4 = tid; i4 < N4; i4 += stride) {
                float4 vx = x4[i4];
                float4 vb = b4[i4];

                // x + b
                vx.x += vb.x; vx.y += vb.y; vx.z += vb.z; vx.w += vb.w;

                // Swish
                float4 vy;
                vy.x = swish_select(vx.x, approx_en_c, approx_mode_c);
                vy.y = swish_select(vx.y, approx_en_c, approx_mode_c);
                vy.z = swish_select(vx.z, approx_en_c, approx_mode_c);
                vy.w = swish_select(vx.w, approx_en_c, approx_mode_c);

                // 可选缩放
                if (use_scale) {
                    vy.x *= scale_c; vy.y *= scale_c; vy.z *= scale_c; vy.w *= scale_c;
                }

                y4[i4] = vy;
            }

            // 尾部
            for (size_t i = rem_base + tid; i < N; i += stride) {
                float v = x[i] + bias_ptr_c[i];
                float r = swish_select(v, approx_en_c, approx_mode_c);
                if (use_scale) r *= scale_c;
                y[i] = r;
            }
            return;
        }

        // 其他不规则偏置形状：回退到标量路径以保证正确性
        // 注意：保持向量化尾部逻辑一致性，使用统一标量回退处理所有元素
    }

    // 标量回退路径（通用，处理所有形状）
    if (use_bias) {
        if (bias_is_scalar) {
            const float b = bias_ptr_c[0];
            for (size_t i = tid; i < N; i += stride) {
                float v = x[i] + b;
                float r = swish_select(v, approx_en_c, approx_mode_c);
                if (use_scale) r *= scale_c;
                y[i] = r;
            }
        } else if (bias_matches_N) {
            for (size_t i = tid; i < N; i += stride) {
                float v = x[i] + bias_ptr_c[i];
                float r = swish_select(v, approx_en_c, approx_mode_c);
                if (use_scale) r *= scale_c;
                y[i] = r;
            }
        } else {
            // 长度为 bias_len_c 的循环广播：bias 索引 = i % bias_len_c
            // 对于大多数情况，这个分支虽然包含取模，但可确保正确性和通用性
            const size_t bl = static_cast<size_t>(bias_len_c);
            for (size_t i = tid; i < N; i += stride) {
                float b = bias_ptr_c[i % bl];
                float v = x[i] + b;
                float r = swish_select(v, approx_en_c, approx_mode_c);
                if (use_scale) r *= scale_c;
                y[i] = r;
            }
        }
    } else {
        // 仅缩放：y = scale * Swish(x)
        for (size_t i = tid; i < N; i += stride) {
            float r = swish_select(x[i], approx_en_c, approx_mode_c);
            y[i] = use_scale ? (r * scale_c) : r;
        }
    }
}

// C++ Wrapper 实现
torch::Tensor kb_25_Swish_wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "kb_25_Swish_wrapper: input must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "kb_25_Swish_wrapper: only float32 is supported");

    // 确保连续性
    auto x = arg0.contiguous();

    // 分配输出张量，形状与输入一致
    auto y = at::empty_like(x);

    // 元素总数
    const size_t N = static_cast<size_t>(x.numel());

    // 指针
    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    // 计算网格/块维度
    const int threads = 256;
    size_t blocks_needed = (N + threads - 1) / threads;
    // 为兼容性限制最大 block 数（gridDim.x）
    const int max_blocks = 65535;
    const int blocks = static_cast<int>(blocks_needed > static_cast<size_t>(max_blocks) ? max_blocks : blocks_needed);

    // 获取当前 CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 调用内核
    if (N > 0) {
        kb_25_Swish_kernel<<<blocks, threads, 0, stream>>>(x_ptr, y_ptr, N);
    }

    // 检查内核错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kb_25_Swish_wrapper: kernel launch failed with error: ", cudaGetErrorString(err));

    return y;
}