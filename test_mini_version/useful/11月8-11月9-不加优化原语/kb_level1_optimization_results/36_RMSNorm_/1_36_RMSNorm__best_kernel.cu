#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_36_RMSNorm__wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

// CUDA 核函数：对 dim=1(通道维) 执行 RMSNorm
// 输入/输出按 [N, C, S] 视图处理，其中 S 是从第 3 维开始的展平大小
__global__ void rmsnorm_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long N,
    long long C,
    long long S,
    float eps
) {
    // 每个 block 处理一个 (n, s) 的 tile，tile 大小为 T=blockDim.x
    const int T = blockDim.x;
    const long long num_s_tiles = (S + T - 1) / T;         // S 维上的 tile 数
    const long long total_tiles = N * num_s_tiles;         // 总 tile 数

    // 使用动态共享内存缓冲（half 类型以扩大容量），由外部在 kernel 启动时提供大小
    extern __shared__ __half s_tile[];

    // 若 S==0，则 num_s_tiles==0，直接退出循环
    for (long long tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {
        // 计算该 tile 对应的 n 和 s 范围
        long long n = (num_s_tiles > 0) ? (tile_id / num_s_tiles) : 0;
        long long s_tile_start = (num_s_tiles > 0) ? ((tile_id % num_s_tiles) * T) : 0;

        // 有效的 tile 大小（处理末尾不足 T 的情况）
        int valid_T = 0;
        if (s_tile_start < S) {
            long long remain = S - s_tile_start;
            valid_T = (remain >= T) ? T : static_cast<int>(remain);
        } else {
            continue;
        }

        int tid = threadIdx.x;
        bool active = (tid < valid_T);

        long long s_base = n * C * S;

        // 只有在共享内存容量足够时使用共享内存路径（使用 half 共享缓冲以扩大容量）
        bool use_shared = (static_cast<long long>(T) * static_cast<long long>(C)) <= 16384LL;

        if (use_shared) {
            // 共同加载：按 [s(线程), c] 的转置布局将该 tile 的数据加载到共享内存（转换为 half）
            for (long long c = 0; c < C; ++c) {
                long long chan_base = s_base + c * S + s_tile_start;
                if (active) {
                    int idx = tid + static_cast<int>(c) * T;
                    s_tile[idx] = __float2half(x[chan_base + tid]);
                }
            }
            __syncthreads();

            if (active) {
                // 每线程计算该 s 位点跨 C 的平方均值（在 float 中累加以保证数值稳定性）
                float sumsq = 0.0f;
                for (long long c = 0; c < C; ++c) {
                    int idx = tid + static_cast<int>(c) * T;
                    float v = __half2float(s_tile[idx]);
                    sumsq += v * v;
                }
                float mean = sumsq / static_cast<float>(C);
                float denom = sqrtf(mean + eps);
                float inv = 1.0f / denom;
                __half inv_h = __float2half(inv);

                // 就地缩放共享内存中的数据（half 乘法）
                for (long long c = 0; c < C; ++c) {
                    int idx = tid + static_cast<int>(c) * T;
                    s_tile[idx] = __hmul(s_tile[idx], inv_h);
                }
            }
            __syncthreads();

            // 共同写回：沿 S 维写回归一化结果
            if (active) {
                long long local_s = s_tile_start + tid;
                for (long long c = 0; c < C; ++c) {
                    long long off = s_base + c * S + local_s;
                    int idx = tid + static_cast<int>(c) * T;
                    y[off] = __half2float(s_tile[idx]);
                }
            }
        } else {
            // 回退路径：不使用共享内存，按原始实现两次遍历 C（一次读求和，一次写缩放）
            if (!active) {
                continue;
            }

            long long local_s = s_tile_start + tid;

            // 计算该线程负责的单个 s 的沿 C 的平方和
            float sumsq = 0.0f;
            for (long long c = 0; c < C; ++c) {
                long long off = s_base + c * S + local_s;
                float v = x[off];
                sumsq += v * v;
            }

            float mean = sumsq / static_cast<float>(C);
            float denom = sqrtf(mean + eps);
            float inv = 1.0f / denom;

            // 归一化写出：再次遍历 C，进行连续访问
            for (long long c = 0; c < C; ++c) {
                long long off = s_base + c * S + local_s;
                y[off] = x[off] * inv;
            }
        }
    }
}

// C++ Wrapper 实现
torch::Tensor kb_1_36_RMSNorm__wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "kb_1_36_RMSNorm__wrapper: input must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "kb_1_36_RMSNorm__wrapper: only float32 is supported");
    TORCH_CHECK(arg0.dim() >= 2, "kb_1_36_RMSNorm__wrapper: input must have at least 2 dimensions");

    // 保证使用正确的设备
    c10::cuda::CUDAGuard device_guard(arg0.device());

    // 保证内存连续
    auto x = arg0.contiguous();

    // 视作 [N, C, S]，S 是从第 3 维开始的展平
    const long long N = x.size(0);
    const long long C = x.size(1);
    TORCH_CHECK(C > 0, "kb_1_36_RMSNorm__wrapper: channel dimension (dim=1) must be > 0");

    const long long total_elems = x.numel();
    TORCH_CHECK(total_elems % (N * C) == 0, "kb_1_36_RMSNorm__wrapper: invalid shape for flattening spatial dims");
    const long long S = total_elems / (N * C);

    auto y = at::empty_like(x);

    // 计算网格/线程配置
    const long long M = N * S;
    int threads = 256;
    // 避免 blocks 为 0
    int blocks = static_cast<int>((M + threads - 1) / threads);
    if (blocks <= 0) blocks = 1;

    constexpr float eps = 1e-5f; // 与给定 PyTorch 实现的默认 eps 一致

    // 当前 CUDA 流
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 动态共享内存大小（仅在 C<=64 时启用 32KB 共享缓冲；否则为 0 以避免共享内存限制）
    size_t shared_bytes = (C <= 64 ? static_cast<size_t>(256LL * C * sizeof(__half)) : 0ULL);

    // 启动 kernel
    rmsnorm_dim1_kernel<<<blocks, threads, shared_bytes, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, S, eps
    );

    // 可选调试同步（生产环境关闭以避免性能损失）
    // cudaStreamSynchronize(stream);

    // 错误检查
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kb_1_36_RMSNorm__wrapper kernel launch failed: ", cudaGetErrorString(err));

    return y;
}