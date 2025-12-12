#pragma once
#include <torch/extension.h>

// C++ Wrapper 函数声明 (保持不变)
torch::Tensor kb_90_cumprod_wrapper(torch::Tensor arg0, int64_t arg1);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>

// -----------------------------------------------------------------------------
//  PyTorch ≥ 2.1 getCurrentCUDAStream 头文件
// -----------------------------------------------------------------------------
#include <ATen/cuda/CUDAContext.h>

// -----------------------------------------------------------------------------
// CUDA error-check 工具
//   说明: 必须在首次使用 *之前* 就完成宏 / 函数定义，否则 NVCC 会出现
//         “identifier … is undefined” 的编译错误。
// -----------------------------------------------------------------------------
#define CHECK_CUDA(err)                                                                 \
    if (err != cudaSuccess) {                                                           \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));  \
        exit(-1);                                                                       \
    }

inline void CUDA_CHECK_ERRORS() {
    cudaError_t err = cudaGetLastError();
    CHECK_CUDA(err);
    err = cudaDeviceSynchronize();
    CHECK_CUDA(err);
}

// -----------------------------------------------------------------------------
// (可选) 示例辅助函数：块级加法归约 —— 这里只做示范，当前内核未使用
// -----------------------------------------------------------------------------
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    // warp 内归约
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 每个 warp 的 lane==0 线程写共享内存
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // 第 0 个 warp 做最终归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// -----------------------------------------------------------------------------
// Tiling / CTA 配置常量
// -----------------------------------------------------------------------------
constexpr int TILE_COLS            = 32;          // 每个 tile 32 列（一个半 warp）保证连续读
constexpr int BLOCK_ROWS           = 8;           // 一个 CTA 负责 8 行
constexpr int THREADS_PER_BLOCK    = TILE_COLS * BLOCK_ROWS;   // 256 线程 / block

// -----------------------------------------------------------------------------
// CUDA kernels
// 说明:
//   1. 目前仅针对 2-D Tensor (N, M)
//   2. dim == 1 : 沿列方向 (对每一行做累积乘法)
//      dim == 0 : 沿行方向 (对每一列做累积乘法)
// -----------------------------------------------------------------------------

// ---------------- Naive dim==1 kernel (保留作参考，当前 wrapper 不再调用) -------------
template <typename scalar_t>
__global__ void cumprod_dim1_kernel_naive(     // dim == 1  (行 → 列累积)
    const scalar_t* __restrict__ input,
    scalar_t*       __restrict__ output,
    int64_t N, int64_t M)
{
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    scalar_t acc = static_cast<scalar_t>(1);
    int64_t base = row * M;
    for (int64_t col = 0; col < M; ++col) {
        int64_t idx = base + col;
        acc *= input[idx];
        output[idx] = acc;
    }
}

// ---------------- 新的 tiled kernel (dim == 1) -----------------------------------
template <typename scalar_t>
__global__ void cumprod_dim1_tiled_kernel(
    const scalar_t* __restrict__ in,
    scalar_t*       __restrict__ out,
    int64_t N, int64_t M)
{
    // Fast lane/warp bookkeeping
    const int lane       = threadIdx.x;   // 0 .. 31
    const int warp_row   = threadIdx.y;   // 0 .. (BLOCK_ROWS-1)
    const int global_row = blockIdx.x * BLOCK_ROWS + warp_row;

    // Early exit for out-of-range rows (entire warp exits coherently)
    if (global_row >= N) return;

    scalar_t row_carry = static_cast<scalar_t>(1);   // carry between tiles

    // Tile loop – each iteration processes TILE_COLS consecutive elements
    for (int base_col = 0; base_col < M; base_col += TILE_COLS) {

        const int gcol      = base_col + lane;
        const bool valid_el = (gcol < M);

        // 3.0 — Load element (or neutral element if out-of-bounds)
        scalar_t val = valid_el ? in[global_row * M + gcol]
                                : static_cast<scalar_t>(1);

        // 3.1 — Inclusive scan inside the warp (multiplicative Hillis–Steele)
        #pragma unroll
        for (int offset = 1; offset < warpSize; offset <<= 1) {
            scalar_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
            if (lane >= offset) val *= n;
        }

        // 3.2 — Write prefix product (carry * val) back to global memory
        if (valid_el) {
            out[global_row * M + gcol] = row_carry * val;
        }

        // 3.3 — Update row_carry with the product of the entire tile
        const int valid_cnt   = min(TILE_COLS, static_cast<int>(M - base_col)); // 1 .. 32
        scalar_t tile_total   = __shfl_sync(0xFFFFFFFF, val, valid_cnt - 1);
        row_carry *= tile_total;
    }
}

// ---------------- 原有 dim==0 kernel（保持不变） -------------------------------------
template <typename scalar_t>
__global__ void cumprod_dim0_kernel(     // dim == 0  (列 → 行累积)
    const scalar_t* __restrict__ input,
    scalar_t*       __restrict__ output,
    int64_t N, int64_t M)
{
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= M) return;

    scalar_t acc = static_cast<scalar_t>(1);
    for (int64_t row = 0; row < N; ++row) {
        int64_t idx = row * M + col;
        acc *= input[idx];
        output[idx] = acc;
    }
}

// -----------------------------------------------------------------------------
// C++ wrapper
// -----------------------------------------------------------------------------
torch::Tensor kb_90_cumprod_wrapper(torch::Tensor arg0, int64_t arg1) {
    // -------------------- 校验 --------------------
    TORCH_CHECK(arg0.is_cuda(),        "Input tensor must be on CUDA device");
    TORCH_CHECK(arg0.is_contiguous(),  "Input tensor must be contiguous");
    TORCH_CHECK(arg0.dim() == 2,       "Only 2-D tensors are supported (got ",
                                       arg0.dim(), "-D)");
    TORCH_CHECK(arg1 == 0 || arg1 == 1,"dim must be 0 or 1 (got ", arg1, ")");

    const int64_t N = arg0.size(0);
    const int64_t M = arg0.size(1);

    auto output = torch::empty_like(arg0);

    auto stream = at::cuda::getCurrentCUDAStream();

    // -------------------- 调度 --------------------
    AT_DISPATCH_FLOATING_TYPES(arg0.scalar_type(), "kb_90_cumprod_cuda", ([&] {
        if (arg1 == 1) {          // 沿 dim==1 计算
            constexpr dim3 threads(TILE_COLS, BLOCK_ROWS);        // (32, 8) = 256 threads
            dim3 blocks((N + BLOCK_ROWS - 1) / BLOCK_ROWS);       // 每个 CTA 处理 BLOCK_ROWS 行
            cumprod_dim1_tiled_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                arg0.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                N, M);
        } else {                  // 沿 dim==0 计算
            constexpr int threads = 64;                           // 沿 dim==0 保持原 block size＝64
            dim3 blocks((M + threads - 1) / threads);
            cumprod_dim0_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                arg0.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                N, M);
        }
    }));

    // -------------------- 错误检查 --------------------
    CUDA_CHECK_ERRORS();
    return output;
}