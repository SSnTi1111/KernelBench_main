#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_91_cumsum_reverse_wrapper(torch::Tensor arg0, int64_t arg1);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

// ============================================================================
//  高性能 2-D 反向 cumulative-sum CUDA 实现
//   • 线程划分 : 每个 warp 负责 1 条扫描线
//   • 块内可容纳若干 warp (blockDim.x / 32)
//   • 仅修改 kernel, wrapper 保持不变
// ============================================================================

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// 默认仍以 256 线程 / block 为编译假设, 与 wrapper 中的配置保持一致
#define BLOCK_THREADS 256

// ---------------------------------------------------------------------------
// 内联帮助函数
// ---------------------------------------------------------------------------
__device__ __forceinline__ int lane_id()
{
    int id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(id));
    return id;
}

/* 反向(右→左 / 下→上) warp-内 inclusive-scan
   ─ 对 lane k 给出 lanes[k, 31] 的和 */
__device__ __forceinline__ float warp_reverse_inclusive_scan(float v)
{
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float tmp = __shfl_down_sync(0xffffffff, v, offset);
        if (lane_id() + offset < WARP_SIZE)
            v += tmp;
    }
    return v;
}

// ---------------------------------------------------------------------------
// kernel-dim1 : 处理行方向 (从最后一列向第一列扫描)
//               每个 warp 负责一行; block 内可含任意数量的 warp
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void reverse_cumsum_dim1_kernel(
        const float* __restrict__ in,
        float*       __restrict__ out,
        int rows,
        int cols)
{
    const int warp_id_in_block = threadIdx.x >> 5;           // 本 block 内 warp 索引
    const int lane             = threadIdx.x & 31;           // 0‥31
    const int warps_per_block  = blockDim.x >> 5;            // 动态计算
    const int row              = blockIdx.x * warps_per_block + warp_id_in_block;

    if (row >= rows) return;

    const int tiles = (cols + WARP_SIZE - 1) / WARP_SIZE;
    float carry = 0.f;

    // 从最右侧 tile 开始, 逐 tile 向左扫描
    for (int t = tiles - 1; t >= 0; --t) {
        const int col = t * WARP_SIZE + lane;

        // 条件化读取
        float val = (col < cols) ? in[row * cols + col] : 0.f;

        // warp 内反向前缀和
        float scan   = warp_reverse_inclusive_scan(val);
        float prefix = scan + carry;

        // 条件化写回
        if (col < cols)
            out[row * cols + col] = prefix;

        // 更新上一 tile 的 carry, 仅 lane0 负责
        if (lane == 0)
            carry += scan;   // lane0 的 scan 已含本 tile 全部和
    }
}

// ---------------------------------------------------------------------------
// kernel-dim0 : 处理列方向 (从最后一行向第一行扫描)
//               每个 warp 负责一列
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void reverse_cumsum_dim0_kernel(
        const float* __restrict__ in,
        float*       __restrict__ out,
        int rows,
        int cols)
{
    const int warp_id_in_block = threadIdx.x >> 5;           // 本 block 内 warp 索引
    const int lane             = threadIdx.x & 31;           // 0‥31
    const int warps_per_block  = blockDim.x >> 5;            // 动态计算
    const int col              = blockIdx.x * warps_per_block + warp_id_in_block;

    if (col >= cols) return;

    const int tiles = (rows + WARP_SIZE - 1) / WARP_SIZE;
    float carry = 0.f;

    for (int t = tiles - 1; t >= 0; --t) {
        const int row = t * WARP_SIZE + lane;

        float val = (row < rows) ? in[row * cols + col] : 0.f;

        float scan   = warp_reverse_inclusive_scan(val);
        float prefix = scan + carry;

        if (row < rows)
            out[row * cols + col] = prefix;

        if (lane == 0)
            carry += scan;
    }
}

// ---------------------------------------------------------------------------
// C++ Wrapper (保持不变)
// ---------------------------------------------------------------------------
torch::Tensor kb_91_cumsum_reverse_wrapper(torch::Tensor arg0, int64_t arg1)
{
    TORCH_CHECK(arg0.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat,
                "Only float32 tensors are supported");
    TORCH_CHECK(arg0.dim() == 2,
                "This reference implementation only supports 2-D tensors");

    // 处理 dim 为负数的情况
    int64_t dim = arg1;
    if (dim < 0) dim += arg0.dim();
    TORCH_CHECK(dim == 0 || dim == 1,
                "dim must be 0 or 1 for 2-D tensor");

    // 保证连续性
    if (!arg0.is_contiguous()) {
        arg0 = arg0.contiguous();
    }

    // 输出张量
    auto out = at::empty_like(arg0);

    const int rows = static_cast<int>(arg0.size(0));
    const int cols = static_cast<int>(arg0.size(1));

    const int threads = 256;
    dim3 blocks;

    // 调用对应 kernel
    const float* in_ptr  = arg0.data_ptr<float>();
    float*       out_ptr = out.data_ptr<float>();

    if (dim == 1) {
        blocks = dim3((rows + threads - 1) / threads);
        reverse_cumsum_dim1_kernel<<<blocks, threads, 0,
            at::cuda::getCurrentCUDAStream()>>>(
                in_ptr, out_ptr, rows, cols);
    } else { // dim == 0
        blocks = dim3((cols + threads - 1) / threads);
        reverse_cumsum_dim0_kernel<<<blocks, threads, 0,
            at::cuda::getCurrentCUDAStream()>>>(
                in_ptr, out_ptr, rows, cols);
    }

    // CUDA 错误检查（调试用）
#ifndef NDEBUG
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ",
                cudaGetErrorString(err));
#endif

    return out;
}