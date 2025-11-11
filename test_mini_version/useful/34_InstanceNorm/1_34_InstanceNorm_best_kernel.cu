#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_34_InstanceNorm_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

// 设备端块归约：对一个block内的float值求和
// 需要调用方提供 shared 指针，大小至少为 blockDim.x / warpSize
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    // warp 内归约
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
    }

    // 每个 warp 的 lane 0 写到共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 仅第一个 warp 进行最终归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
        }
    }
    return val;
}

// InstanceNorm2d 内核：对每个 (n, c) 平面做归一化
// 输入输出均为 NCHW, float32
__global__ void instance_norm2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    float eps
) {
    int nc = blockIdx.x;  // 0..N*C-1
    int n = nc / C;
    int c = nc % C;

    const int HW = H * W;
    if (HW == 0) return;

    const long long plane_size = static_cast<long long>(HW);
    const long long base = (static_cast<long long>(n) * C + c) * plane_size;

    extern __shared__ float s_red[]; // size = num_warps
    __shared__ float s_mean;
    __shared__ float s_invstd;

    // 单次遍历中缓存每个线程前若干个元素到寄存器，避免二次全局内存读取
    // 选择较小的缓存深度以控制寄存器压力
    constexpr int MAX_CACHE_PER_THREAD = 8;
    float cache_vals[MAX_CACHE_PER_THREAD];

    // 统计阶段：累计 sum、sumsq，并在读取时缓存前 MAX_CACHE_PER_THREAD 个元素
    float sum = 0.0f;
    float sumsq = 0.0f;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    int stored = 0; // 实际缓存的元素个数（不超过 MAX_CACHE_PER_THREAD）
    int total_iters = 0; // 当前线程负责的元素总数

    // 单次读取全局内存，统计并缓存
    for (int idx = tid; idx < HW; idx += stride) {
        float v = x[base + idx];
        sum   += v;
        sumsq += v * v;

        if (stored < MAX_CACHE_PER_THREAD) {
            cache_vals[stored] = v;
            ++stored;
        }
        ++total_iters;
    }

    // 块内归约得到该 (n,c) 平面的 sum 与 sumsq
    float red_sum   = blockReduceSum(sum, s_red);
    float red_sumsq = blockReduceSum(sumsq, s_red);

    if (threadIdx.x == 0) {
        float mean = red_sum / static_cast<float>(HW);
        float var  = red_sumsq / static_cast<float>(HW) - mean * mean;
        s_mean   = mean;
        s_invstd = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float invstd = s_invstd;

    // 归一化写回阶段：
    // 1) 若该线程负责的元素数量不超过缓存容量，则直接使用寄存器缓存的值进行写回，避免再次读取全局内存
    // 2) 若超过缓存容量，则先写回已缓存的部分，再对剩余元素做一次回访读取并写回（回退路径）
    if (total_iters <= MAX_CACHE_PER_THREAD) {
        // 全部使用缓存写回
        int i = 0;
        int out_idx = tid;
        while (i < stored && out_idx < HW) {
            float v = cache_vals[i];
            y[base + out_idx] = (v - mean) * invstd;
            ++i;
            out_idx += stride;
        }
    } else {
        // 先写回已缓存部分
        int i = 0;
        int out_idx = tid;
        while (i < stored && out_idx < HW) {
            float v = cache_vals[i];
            y[base + out_idx] = (v - mean) * invstd;
            ++i;
            out_idx += stride;
        }
        // 对于未缓存部分，回退读取一次（仅限该线程负责的剩余元素）
        // 第一个未缓存的元素对应第 stored 次迭代
        int k = stored;
        for (int idx = tid + k * stride; idx < HW; idx += stride, ++k) {
            float v = x[base + idx];
            y[base + idx] = (v - mean) * invstd;
        }
    }
}

// C++ Wrapper 实现
torch::Tensor kb_1_34_InstanceNorm_wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "arg0 must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "arg0 must be float32");
    TORCH_CHECK(arg0.dim() == 4, "Expected 4D tensor in NCHW layout");

    auto x = arg0.contiguous();

    const int64_t N64 = x.size(0);
    const int64_t C64 = x.size(1);
    const int64_t H64 = x.size(2);
    const int64_t W64 = x.size(3);

    TORCH_CHECK(N64 > 0 && C64 > 0 && H64 > 0 && W64 > 0, "Invalid tensor sizes");

    // 转为 int（这些维度通常小于 2^31）
    int N = static_cast<int>(N64);
    int C = static_cast<int>(C64);
    int H = static_cast<int>(H64);
    int W = static_cast<int>(W64);

    auto y = torch::empty_like(x);

    const int blocks = N * C;
    const int threads = 256; // 8 warps
    const size_t shmem_bytes = (threads / 32) * sizeof(float); // 每个warp一个槽位

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const float eps = 1e-5f; // 与 PyTorch 默认一致

    instance_norm2d_kernel<<<blocks, threads, shmem_bytes, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        eps
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return y;
}