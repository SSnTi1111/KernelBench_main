#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_37_FrobeniusNorm__wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
// PyTorch 2.1+ 移除了 c10::cuda::getCurrentCUDAStream
// 使用 at::cuda::getCurrentCUDAStream() 代替
#include <ATen/cuda/CUDAContext.h>

// ------------- CUDA 辅助定义 -------------
// 显式定义 warpSize 以避免链接错误
#ifndef warpSize
#define warpSize 32
#endif

// ------------- CUDA 辅助函数 (在 kernel 之前定义) -------------

// Warp 内归约
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block 内归约，shared 需至少提供 ceil(blockDim.x / warpSize) 个 float
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    // 先做 warp 内归约
    val = warpReduceSum(val);

    // 每个 warp 的 lane0 将结果写入 shared
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 仅使用第一个 warp 对所有 warp 的部分和做最终归约
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    float block_sum = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) {
        block_sum = warpReduceSum(block_sum);
    }
    return block_sum;
}

// ------------- CUDA 内核实现 -------------

// 计算全局 Frobenius 范数的平方：sum(x^2)
// sum_out 是单元素 device 内存，累加到其中
__global__ void reduce_sum_squares_kernel(const float* __restrict__ x,
                                          size_t N,
                                          float* __restrict__ sum_out) {
    extern __shared__ float shared[]; // 用于块内归约（大小为 num_warps）
    float local = 0.0f;

    // 以 tile 为单位的 4x 展开：每个 tile 覆盖 blockDim.x * 4 个元素
    const size_t B = (size_t)blockDim.x;
    const size_t G = (size_t)gridDim.x;
    const size_t tile_span = B * 4;               // 每个 tile 的跨度
    const size_t grid_tile_stride = G * tile_span; // 相邻迭代的 tile 起点间隔

    // 每个线程在 tile 内部处理 4 个以 blockDim.x 为步长的元素：
    // base, base + B, base + 2B, base + 3B
    size_t base = (size_t)blockIdx.x * tile_span + threadIdx.x;

    for (size_t tile_base = base; tile_base < N; tile_base += grid_tile_stride) {
        size_t j0 = tile_base;
        size_t j1 = j0 + B;
        size_t j2 = j1 + B;
        size_t j3 = j2 + B;

        if (j0 < N) {
            float v0 = __ldg(x + j0);
            local = fmaf(v0, v0, local);
        }
        if (j1 < N) {
            float v1 = __ldg(x + j1);
            local = fmaf(v1, v1, local);
        }
        if (j2 < N) {
            float v2 = __ldg(x + j2);
            local = fmaf(v2, v2, local);
        }
        if (j3 < N) {
            float v3 = __ldg(x + j3);
            local = fmaf(v3, v3, local);
        }
    }

    // 块内归约到单个值
    float block_sum = blockReduceSum(local, shared);

    // 仅线程0将块内结果原子加到全局
    if (threadIdx.x == 0) {
        atomicAdd(sum_out, block_sum);
    }
}

// 使用 sum_ptr 指向的范数平方，计算 1/sqrt(sum) 并对每个元素归一化
__global__ void normalize_with_sum_kernel(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          size_t N,
                                          const float* __restrict__ sum_ptr) {
    // 读取一次全局标量
    float sumsq = __ldg(sum_ptr);
    // 使用 rsqrtf 提高性能（当 sumsq == 0 时返回 +inf，行为与 x / 0 一致，得到 inf 或 NaN）
    float inv_norm = rsqrtf(sumsq);

    // 本地常量定义
    const size_t B = (size_t)blockDim.x;
    const size_t tile_span = B * 4ULL;                  // 每个 tile 覆盖 4*B 个元素
    const size_t grid_tile_stride = (size_t)gridDim.x * tile_span;

    // 为 tile 提供共享内存缓存（4*B == 1024 个 float，B=256）
    __shared__ float s_tile[1024];

    // 以 block 为单位的 grid-stride tile 循环
    for (size_t tile_base = (size_t)blockIdx.x * tile_span;
         tile_base < N;
         tile_base += grid_tile_stride) {

        // 协作加载阶段：每个线程加载 4 个元素到共享内存，保证合并访问
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            size_t gidx = tile_base + (size_t)threadIdx.x + (size_t)k * B;
            size_t sidx = (size_t)threadIdx.x + (size_t)k * B;
            float v = 0.0f;
            if (gidx < N) {
                v = __ldg(x + gidx);
            }
            s_tile[sidx] = v;
        }
        __syncthreads();

        // 计算与写回阶段：从共享内存读取，乘以 inv_norm 并写回全局
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            size_t sidx = (size_t)threadIdx.x + (size_t)k * B;
            size_t gidx = tile_base + sidx;
            if (gidx < N) {
                y[gidx] = s_tile[sidx] * inv_norm;
            }
        }

        // 确保本 tile 的共享内存使用完成再进入下一 tile 的加载
        __syncthreads();
    }
}

// ------------- C++ Wrapper 实现 -------------

torch::Tensor kb_1_37_FrobeniusNorm__wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "kb_1_37_FrobeniusNorm__wrapper: input must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "kb_1_37_FrobeniusNorm__wrapper: only float32 is supported");

    // 确保连续
    auto x = arg0.contiguous();

    // 元素总数
    const size_t N = static_cast<size_t>(x.numel());
    // 如果为空张量，按 PyTorch 语义返回相同形状的张量（与 x / norm 一致，norm 为 0，结果仍为空）
    if (N == 0) {
        return x.clone();
    }

    auto options = x.options();
    auto y = torch::empty_like(x);

    // 在同一设备上创建一个单元素张量用于保存 sum(x^2)
    auto sum_tensor = torch::empty({1}, options);
    float* d_sum = sum_tensor.data_ptr<float>();

    // 获取当前 CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 将 sum 初始化为 0
    cudaError_t err = cudaMemsetAsync(d_sum, 0, sizeof(float), stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemsetAsync failed: ", cudaGetErrorString(err));

    // 配置 kernel
    const int block = 256;
    // 1D grid 上限（CUDA 计算能力允许的最大 gridDim.x，一般 2^31-1，但为兼容性用 65535）
    const unsigned long long max_grid = 65535ULL;
    unsigned long long needed_blocks = (N + (unsigned long long)block - 1ULL) / (unsigned long long)block;
    int grid = static_cast<int>(std::min(needed_blocks, max_grid));

    // 动态共享内存大小（每个 warp 一个 float）
    int num_warps = (block + warpSize - 1) / warpSize;
    size_t shared_bytes = static_cast<size_t>(num_warps) * sizeof(float);

    // 启动归约 kernel
    reduce_sum_squares_kernel<<<grid, block, shared_bytes, stream>>>(
        x.data_ptr<float>(), N, d_sum
    );
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "reduce_sum_squares_kernel launch failed: ", cudaGetErrorString(err));

    // 归一化 kernel：读取 sum 并对每个元素做除法
    normalize_with_sum_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), N, d_sum
    );
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "normalize_with_sum_kernel launch failed: ", cudaGetErrorString(err));

    return y;
}