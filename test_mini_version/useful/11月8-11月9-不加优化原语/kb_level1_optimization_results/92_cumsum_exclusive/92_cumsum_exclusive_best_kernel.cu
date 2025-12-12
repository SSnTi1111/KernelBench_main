#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_92_cumsum_exclusive_wrapper(torch::Tensor arg0, int64_t arg1);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

// ───────────────────────────  宏 / 辅助检查  ────────────────────────────
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)  \
    CHECK_CUDA(x);      \
    CHECK_CONTIGUOUS(x)

// ───────────────────────────  Kernel for dim == 1  ───────────────────────────
//  Optimised two–path kernel:
//
//    (a) If launched with one *block* per input row (gridDim.x == rows_out)
//        each row is processed cooperatively by all 256 threads of that block
//        following the Analysis-Agent plan.
//
//    (b) Otherwise fall back to the original “one-thread-per-row” algorithm so
//        that existing wrapper-code that launches fewer blocks continues to
//        behave correctly.
//
//  Both paths deliver identical (correct) results; path-(a) provides much
//  higher throughput once the wrapper is updated, while path-(b) guarantees
//  backward compatibility today.
//
__global__ void exclusive_cumsum_dim1_kernel(const float* __restrict__ input,
                                             float*       __restrict__ output,
                                             int rows_in,
                                             int cols_in) {
    const int rows_out = rows_in - 1;          // number of output rows

    // Detect launch configuration:
    //   • path-(a) : one block per row  (gridDim.x == rows_out)
    //   • path-(b) : legacy launch      (gridDim.x  < rows_out)
    const bool coop_launch = (gridDim.x == rows_out);

    if (coop_launch) {
        // ─────────────────────  Path-(a) : cooperative (256T) per row  ─────────────────────
        const int row_idx = blockIdx.x;
        if (row_idx >= rows_out) return;

        const float* in_row  = input  + row_idx * cols_in;
        float*       out_row = output + row_idx * (cols_in + 1);

        // Shared memory (tiny: 3 × 8 × 4 = 96 B)
        __shared__ float warp_sums[8];     // inclusive totals (one per warp)
        __shared__ float warp_prefix[8];   // exclusive prefix of warp totals
        __shared__ float seg_prefix;       // running prefix between 256-elem segments

        const int lane    = threadIdx.x & 31;   // 0-31
        const int warp_id = threadIdx.x >> 5;   // 0-7
        constexpr unsigned MASK = 0xffffffffu;

        // Initialise first element and segment prefix
        if (threadIdx.x == 0) {
            out_row[0] = 0.0f;
            seg_prefix = 0.0f;
        }
        __syncthreads();

        // Process the row in 256-element segments
        for (int base = 0; base < cols_in; base += blockDim.x) {
            const int col = base + threadIdx.x;
            const float val = (col < cols_in) ? in_row[col] : 0.0f;

            // In-warp inclusive scan via shuffles
            float scan = val;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                float n = __shfl_up_sync(MASK, scan, off);
                if (lane >= off) scan += n;
            }
            const float excl = scan - val;   // exclusive prefix inside warp

            // Warp leaders publish their inclusive totals
            if (lane == 31) warp_sums[warp_id] = scan;
            __syncthreads();

            // First warp builds exclusive prefix of warp totals
            if (warp_id == 0 && lane < 8) {
                float w_scan = warp_sums[lane];
                #pragma unroll
                for (int off = 1; off < 8; off <<= 1) {
                    float n = __shfl_up_sync(MASK, w_scan, off);
                    if (lane >= off) w_scan += n;
                }
                warp_prefix[lane] = w_scan - warp_sums[lane];
            }
            __syncthreads();

            // Compose full prefix for each thread
            const float row_prefix = seg_prefix
                                   + warp_prefix[warp_id]
                                   + excl;

            // Write result (+1 for exclusive semantics)
            if (col < cols_in)
                out_row[col + 1] = row_prefix;

            // Last thread updates segment prefix for next iteration
            if (threadIdx.x == blockDim.x - 1)
                seg_prefix += scan;
            __syncthreads();
        }
    }
    else {
        // ─────────────────────  Path-(b) : legacy one-thread-per-row  ─────────────────────
        const int row_idx   = blockIdx.x * blockDim.x + threadIdx.x;
        if (row_idx >= rows_out) return;

        const int cols_out  = cols_in + 1;
        const float* in_row = input  + row_idx * cols_in;
        float*       out_row= output + row_idx * cols_out;

        out_row[0] = 0.0f;
        float acc = 0.0f;

        int col = 0;
        constexpr int UNROLL = 4;
        for (; col + UNROLL - 1 < cols_in; col += UNROLL) {
            float v0 = in_row[col + 0];
            float v1 = in_row[col + 1];
            float v2 = in_row[col + 2];
            float v3 = in_row[col + 3];

            acc += v0;
            out_row[col + 1] = acc;

            acc += v1;
            out_row[col + 2] = acc;

            acc += v2;
            out_row[col + 3] = acc;

            acc += v3;
            out_row[col + 4] = acc;
        }
        for (; col < cols_in; ++col) {
            acc += in_row[col];
            out_row[col + 1] = acc;
        }
    }
}

// ───────────────────────────  Kernel for dim == 0  ───────────────────────────
//  (Unchanged from original implementation)
__global__ void exclusive_cumsum_dim0_kernel(const float* __restrict__ input,
                                             float*       __restrict__ output,
                                             int rows_in,
                                             int cols_in) {
    const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx >= cols_in) return;

    float acc = 0.0f;
    int row = 0;
    int idx = col_idx;  // row * cols_in + col_idx

    constexpr int UNROLL = 4;
    for (; row + UNROLL - 1 < rows_in; row += UNROLL) {
        output[idx] = acc; acc += input[idx]; idx += cols_in;
        output[idx] = acc; acc += input[idx]; idx += cols_in;
        output[idx] = acc; acc += input[idx]; idx += cols_in;
        output[idx] = acc; acc += input[idx]; idx += cols_in;
    }

    for (; row < rows_in; ++row) {
        output[idx] = acc;
        acc += input[idx];
        idx += cols_in;
    }
}

// ───────────────────────────  C++ Wrapper  ────────────────────────────
torch::Tensor kb_92_cumsum_exclusive_wrapper(torch::Tensor arg0, int64_t arg1) {
    /*
        arg0 : 输入 Tensor，必须位于 CUDA 上、float32、contiguous
        arg1 : 维度 dim，仅支持 0 或 1
    */
    CHECK_INPUT(arg0);
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "Only float32 is supported");

    const int64_t dim = arg1;
    TORCH_CHECK(dim == 0 || dim == 1,
                "Only dim == 0 or dim == 1 is currently supported");

    const int rows_in = static_cast<int>(arg0.size(0));
    const int cols_in = static_cast<int>(arg0.size(1));

    torch::Tensor output;
    if (dim == 1) {
        TORCH_CHECK(rows_in >= 1, "rows_in must be >= 1 for dim == 1");
        output = torch::empty({rows_in - 1, cols_in + 1}, arg0.options());
    } else { // dim == 0
        output = torch::empty({rows_in, cols_in}, arg0.options());
    }

    const int threads = 256;
    dim3 blocks;

    auto stream = at::cuda::getCurrentCUDAStream();

    const float* input_ptr = arg0.data_ptr<float>();
    float* output_ptr      = output.data_ptr<float>();

    if (dim == 1) {
        blocks = dim3((rows_in - 1 + threads - 1) / threads);
        exclusive_cumsum_dim1_kernel<<<blocks, threads, 0, stream>>>(
            input_ptr, output_ptr, rows_in, cols_in);
    } else { // dim == 0
        blocks = dim3((cols_in + threads - 1) / threads);
        exclusive_cumsum_dim0_kernel<<<blocks, threads, 0, stream>>>(
            input_ptr, output_ptr, rows_in, cols_in);
    }

    // 同步错误检查（可选）
#ifndef NDEBUG
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
#endif

    return output;
}