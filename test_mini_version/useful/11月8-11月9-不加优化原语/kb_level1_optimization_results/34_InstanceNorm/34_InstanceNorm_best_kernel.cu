#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_34_InstanceNorm_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// CUDA 辅助函数：Warp 内求和
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// CUDA 辅助函数：Block 内求和并广播给所有线程
// shared 的大小至少为 numWarps = ceil(blockDim.x / warpSize)
__device__ float blockReduceSumAll(float val, float* shared) {
    int lane = threadIdx.x & (warpSize - 1);
    int wid  = threadIdx.x >> 5; // / warpSize

    // 先做 warp 内归约
    val = warpReduceSum(val);

    // 每个 warp 的 lane 0 写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 由第一个 warp 对所有 warp 的部分和进行归约
    float res = 0.0f;
    if (wid == 0) {
        // 计算本 block 的 warp 数
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        float v = (lane < numWarps) ? shared[lane] : 0.0f;
        v = warpReduceSum(v);
        if (lane == 0) {
            shared[0] = v; // 将最终结果放到 shared[0]
        }
    }
    __syncthreads();
    // 广播结果给所有线程
    res = shared[0];
    return res;
}

// CUDA 内核：对输入做 InstanceNorm2d（每个 (n,c) 独立在 HxW 上做均值/方差归一化）
__global__ void instance_norm_nchw_kernel(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int N, int C, int H, int W,
                                          float eps) {
    int nc = blockIdx.x; // 每个 block 处理一个 (n, c)
    if (nc >= N * C) return;

    int n = nc / C;
    int c = nc % C;

    int HW = H * W;
    size_t base = (static_cast<size_t>(n) * C + static_cast<size_t>(c)) * static_cast<size_t>(HW);

    extern __shared__ float sshared[]; // 大小为 numWarps 的共享内存

    // 线程本地统计
    float thread_sum = 0.0f;
    float thread_sumsq = 0.0f;

    // 每个线程批处理元素并尽量复用寄存器来避免二次全局内存读取
    constexpr int BATCH = 8; // 每线程寄存器缓存的批大小
    const int capacity = blockDim.x * BATCH;

    if (HW <= capacity) {
        // 快路径：整个 HxW 能在一次批处理中被所有线程寄存器缓存
        float vals[BATCH];
        int idxs[BATCH];

        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            int idx = threadIdx.x + b * blockDim.x;
            if (idx < HW) {
                float v = x[base + static_cast<size_t>(idx)];
                vals[b] = v;
                idxs[b] = idx;
                thread_sum   += v;
                thread_sumsq += v * v;
            } else {
                idxs[b] = -1;
                vals[b] = 0.0f;
            }
        }

        // Block 内归约得到总和与平方和（广播到所有线程）
        float sum   = blockReduceSumAll(thread_sum, sshared);
        float sumsq = blockReduceSumAll(thread_sumsq, sshared);

        float invHW = 1.0f / static_cast<float>(HW);
        float mean = sum * invHW;
        float var = fmaxf(sumsq * invHW - mean * mean, 0.0f);
        float inv_std = rsqrtf(var + eps);

        // 使用寄存器中缓存的值进行归一化并写回
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            int idx = idxs[b];
            if (idx >= 0) {
                float v = vals[b];
                y[base + static_cast<size_t>(idx)] = (v - mean) * inv_std;
            }
        }
        return;
    }

    // 通用路径：分批遍历（两遍），第一遍统计，第二遍归一化
    // 1) 统计 sum 和 sumsq
    for (int start = threadIdx.x; start < HW; start += blockDim.x * BATCH) {
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            int idx = start + b * blockDim.x;
            if (idx < HW) {
                float v = x[base + static_cast<size_t>(idx)];
                thread_sum   += v;
                thread_sumsq += v * v;
            }
        }
    }

    // Block 内归约得到总和与平方和（广播到所有线程）
    float sum   = blockReduceSumAll(thread_sum, sshared);
    float sumsq = blockReduceSumAll(thread_sumsq, sshared);

    float invHW = 1.0f / static_cast<float>(HW);
    float mean = sum * invHW;
    float var = fmaxf(sumsq * invHW - mean * mean, 0.0f);
    float inv_std = rsqrtf(var + eps);

    // 2) 第二遍：归一化并写输出（按批处理确保访存合并）
    for (int start = threadIdx.x; start < HW; start += blockDim.x * BATCH) {
        #pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            int idx = start + b * blockDim.x;
            if (idx < HW) {
                float v = x[base + static_cast<size_t>(idx)];
                y[base + static_cast<size_t>(idx)] = (v - mean) * inv_std;
            }
        }
    }
}

// C++ Wrapper 实现
torch::Tensor kb_34_InstanceNorm_wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "kb_34_InstanceNorm_wrapper: arg0 must be a CUDA tensor");
    TORCH_CHECK(arg0.dtype() == torch::kFloat32, "kb_34_InstanceNorm_wrapper: only float32 is supported");
    TORCH_CHECK(arg0.dim() == 4, "kb_34_InstanceNorm_wrapper: expected 4D NCHW tensor");

    auto x = arg0.contiguous();
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto y = torch::empty_like(x);

    // 配置 CUDA 启动参数
    int threads = 256;
    int64_t blocks64 = N * C;
    TORCH_CHECK(blocks64 > 0, "kb_34_InstanceNorm_wrapper: invalid grid size");
    TORCH_CHECK(blocks64 <= static_cast<int64_t>(std::numeric_limits<int>::max()),
                "kb_34_InstanceNorm_wrapper: grid size exceeds int range");
    int blocks = static_cast<int>(blocks64);

    int numWarps = (threads + 31) / 32;
    size_t shmem = static_cast<size_t>(numWarps) * sizeof(float);

    float eps = 1e-5f;

    auto stream = at::cuda::getCurrentCUDAStream();

    instance_norm_nchw_kernel<<<blocks, threads, shmem, stream.stream()>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        eps
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}