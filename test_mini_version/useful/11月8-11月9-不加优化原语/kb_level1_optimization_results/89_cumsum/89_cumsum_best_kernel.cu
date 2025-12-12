#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_89_cumsum_wrapper(torch::Tensor arg0, int64_t arg1);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

// PyTorch ≥2.1
#include <ATen/cuda/CUDAContext.h>

// -----------------------------------------------------------------------------
//  Tiled, bank-conflict-free transpose kernel (32×8 = 256 threads / block)
// -----------------------------------------------------------------------------
constexpr int BLOCK_DIM  = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_tiled(
        const float* __restrict__ in,
        float*       __restrict__ out,
        int          rows,    // Height  (Y dimension)
        int          cols)    // Width   (X dimension)
{
    __shared__ float tile[BLOCK_DIM][BLOCK_DIM + 1];   // +1 to avoid bank conflicts

    int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int y = blockIdx.y * BLOCK_DIM + threadIdx.y;

    // ── Load tile with coalesced reads ──────────────────────────────────────────
    #pragma unroll
    for (int i = 0; i < BLOCK_DIM; i += BLOCK_ROWS) {
        if (x < cols && (y + i) < rows) {
            tile[threadIdx.y + i][threadIdx.x] = in[(y + i) * cols + x];
        }
    }

    __syncthreads();

    // ── Write transposed tile with coalesced writes ────────────────────────────
    x = blockIdx.y * BLOCK_DIM + threadIdx.x;   // Note swap of x/y
    y = blockIdx.x * BLOCK_DIM + threadIdx.y;

    #pragma unroll
    for (int i = 0; i < BLOCK_DIM; i += BLOCK_ROWS) {
        if (x < rows && (y + i) < cols) {
            out[(y + i) * rows + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

// -----------------------------------------------------------------------------
//  内部辅助：块级前缀扫描 (256-thread block，8 warps)
// -----------------------------------------------------------------------------
template<int BLOCK_THREADS>
__device__ inline float blockInclusiveScan(float v, float* s_data)
{
    const int tid  = threadIdx.x;
    const int lane = tid & 31;          // 0‥31
    const int wid  = tid >> 5;          // 0‥7  (BLOCK_THREADS==256)

    // ── 1. 每线程把自己的值写入 shared ──────────────────────────────────────────────
    s_data[tid] = v;
    __syncthreads();

    // ── 2. Warp-level前缀扫描 ───────────────────────────────────────────────────────
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        float n = __shfl_up_sync(0xffffffff, v, d);
        if (lane >= d) v += n;
    }

    // ── 3. 把每个 warp 的累积和写回 shared ──────────────────────────────────────────
    if (lane == 31) s_data[wid] = v;
    __syncthreads();

    // ── 4. 第 0 个 warp 扫描 8 个 warp 累积和 ───────────────────────────────────────
    if (wid == 0) {
        float warp_sum = (lane < (BLOCK_THREADS >> 5)) ? s_data[lane] : 0.f;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            float n = __shfl_up_sync(0xffffffff, warp_sum, d);
            if (lane >= d) warp_sum += n;
        }
        if (lane < (BLOCK_THREADS >> 5))
            s_data[lane] = warp_sum;
    }
    __syncthreads();

    // ── 5. 各 warp 取前缀并累加 ────────────────────────────────────────────────────
    if (wid > 0) v += s_data[wid - 1];
    return v;
}

// -----------------------------------------------------------------------------
//  向量化前缀和 Kernel
// -----------------------------------------------------------------------------
constexpr int VEC = 4;                // 4×float = 16-B transaction
using Vec4 = float4;

/*
 * 高并行 2-D 累加和（prefix-sum）Kernel（向量化版）
 *
 *   • 每 Block 负责 1 行 (dim==1) 或 1 列 (dim==0)
 *   • 256 线程/Block，每线程一次处理 4 个连续标量
 */
template<int BLOCK_THREADS=256>
__global__ void cumsum_kernel_2d(
        const float* __restrict__ input,
        float*       __restrict__ output,
        int64_t rows,            // size(0)
        int64_t cols,            // size(1)
        int64_t dim)             // 0 或 1
{
    extern __shared__ float s_data[];  // BLOCK_THREADS × sizeof(float)

    const int tid = threadIdx.x;

    if (dim == 1) { // ── 按行扫描 ────────────────────────────────────────────────
        int64_t row = blockIdx.x;
        if (row >= rows) return;

        const int64_t N = cols;
        const float*  ptr_in  = input  + row * cols;
        float*        ptr_out = output + row * cols;

        float carry = 0.f;

        for (int64_t base = 0; base < N; base += BLOCK_THREADS * VEC) {
            // ── 0. 线程是否在当前 Chunk 中有效 ────────────────────────────────
            int64_t elem0 = base + static_cast<int64_t>(tid) * VEC;
            int64_t remain = N - base;
            int valid_threads = static_cast<int>((remain + VEC - 1) / VEC);
            bool thread_active = tid < valid_threads;

            // ── 1. 向量化加载 ────────────────────────────────────────────────
            Vec4 reg_vec{0.f, 0.f, 0.f, 0.f};

            if (thread_active && elem0 + (VEC - 1) < N) {
                reg_vec = reinterpret_cast<const Vec4*>(ptr_in + elem0)[0];
            } else if (thread_active) {  // 尾部
                float tmp[VEC] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    if (elem0 + i < N)
                        tmp[i] = ptr_in[elem0 + i];
                }
                reg_vec = *reinterpret_cast<Vec4*>(tmp);
            }

            float f0 = reg_vec.x;
            float f1 = reg_vec.y;
            float f2 = reg_vec.z;
            float f3 = reg_vec.w;

            // ── 2. 线程内累加 ────────────────────────────────────────────────
            float s0 = f0;
            float s1 = s0 + f1;
            float s2 = s1 + f2;
            float s3 = s2 + f3;
            float thread_total = s3;                 // 本线程 4 元素和

            // ── 3. Block 级扫描获取 prefix_end ───────────────────────────────
            float prefix_end = blockInclusiveScan<BLOCK_THREADS>(thread_total, s_data);

            // ── 4. 计算跨线程 carry ──────────────────────────────────────────
            float carry_prev_threads = carry + (prefix_end - thread_total);

            // ── 5. 输出 4 个前缀值 ──────────────────────────────────────────
            float o0 = carry_prev_threads + s0;
            float o1 = carry_prev_threads + s1;
            float o2 = carry_prev_threads + s2;
            float o3 = carry_prev_threads + s3;

            if (thread_active && elem0 + (VEC - 1) < N) {
                Vec4 out_vec = make_float4(o0, o1, o2, o3);
                reinterpret_cast<Vec4*>(ptr_out + elem0)[0] = out_vec;
            } else if (thread_active) {
                if (elem0     < N) ptr_out[elem0    ] = o0;
                if (elem0 + 1 < N) ptr_out[elem0 + 1] = o1;
                if (elem0 + 2 < N) ptr_out[elem0 + 2] = o2;
                if (elem0 + 3 < N) ptr_out[elem0 + 3] = o3;
            }

            // ── 6. 更新跨 Chunk carry ───────────────────────────────────────
            if (tid == valid_threads - 1)
                s_data[0] = carry + prefix_end;    // 当前 Chunk 总和 + 历史 carry
            __syncthreads();
            if (tid == 0)
                carry = s_data[0];
            __syncthreads();
        }
    } else {        // ── 按列扫描 ────────────────────────────────────────────────
        int64_t col = blockIdx.x;
        if (col >= cols) return;

        const int64_t N = rows;

        float carry = 0.f;

        for (int64_t base = 0; base < N; base += BLOCK_THREADS * VEC) {
            int64_t elem0_row = base + static_cast<int64_t>(tid) * VEC;
            int64_t remain = N - base;
            int valid_threads = static_cast<int>((remain + VEC - 1) / VEC);
            bool thread_active = tid < valid_threads;

            // ── 1. 加载（逐标量，因跨行 stride）──────────────────────────────
            float f[4] = {0.f, 0.f, 0.f, 0.f};
            if (thread_active) {
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    int64_t row_idx = elem0_row + i;
                    if (row_idx < N)
                        f[i] = input[row_idx * cols + col];
                }
            }

            // ── 2. 线程内 prefix ────────────────────────────────────────────
            float s0 = f[0];
            float s1 = s0 + f[1];
            float s2 = s1 + f[2];
            float s3 = s2 + f[3];
            float thread_total = s3;

            float prefix_end = blockInclusiveScan<BLOCK_THREADS>(thread_total, s_data);
            float carry_prev_threads = carry + (prefix_end - thread_total);

            float o0 = carry_prev_threads + s0;
            float o1 = carry_prev_threads + s1;
            float o2 = carry_prev_threads + s2;
            float o3 = carry_prev_threads + s3;

            if (thread_active) {
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    int64_t row_idx = elem0_row + i;
                    if (row_idx < N) {
                        float val = (i == 0) ? o0 :
                                    (i == 1) ? o1 :
                                    (i == 2) ? o2 : o3;
                        output[row_idx * cols + col] = val;
                    }
                }
            }

            // ── 3. 更新跨 Chunk carry ───────────────────────────────────────
            if (tid == valid_threads - 1)
                s_data[0] = carry + prefix_end;
            __syncthreads();
            if (tid == 0)
                carry = s_data[0];
            __syncthreads();
        }
    }
}

// C++ Wrapper 实现
torch::Tensor kb_89_cumsum_wrapper(torch::Tensor arg0, int64_t arg1) {
    TORCH_CHECK(arg0.is_cuda(), "Input tensor must reside on CUDA device");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat,
                "Only float32 tensors are currently supported");
    TORCH_CHECK(arg0.dim() == 2,
                "This reference implementation only supports 2-D tensors");

    // 处理负 dim
    int64_t dim = arg1;
    if (dim < 0) dim += arg0.dim();
    TORCH_CHECK(dim == 0 || dim == 1,
                "Dimension out of range (expected 0 or 1)");

    // 保证连续内存布局
    auto input  = arg0.contiguous();
    auto output = at::empty_like(input);

    const int64_t rows = input.size(0);
    const int64_t cols = input.size(1);

    // 配置 launch 参数
    constexpr int THREADS = 256;
    dim3 block(THREADS);
    dim3 grid;

    // 新策略：一个 Block ↔ 一行 / 一列
    if (dim == 1) {          // 行扫描
        grid.x = rows;
    } else {                 // 列扫描
        grid.x = cols;
    }

    // 获取当前 CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Kernel launch（动态 shared memory = 1KB）
    size_t shmem = THREADS * sizeof(float);
    cumsum_kernel_2d<<<grid, block, shmem, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols,
        dim
    );

    return output;
}