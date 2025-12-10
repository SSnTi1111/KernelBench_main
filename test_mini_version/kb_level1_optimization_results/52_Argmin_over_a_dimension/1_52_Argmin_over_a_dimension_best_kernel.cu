#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_52_Argmin_over_a_dimension_wrapper(torch::Tensor arg0, int64_t arg1);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <cstdint>

// PyTorch ≥2.1 需使用 at::cuda::getCurrentCUDAStream()
#include <ATen/cuda/CUDAContext.h>

/*
 * ---------------------------------------------------------------------------
 * （可选）CUDA 辅助函数示例（此示例未在当前 kernel 中使用，保留作模板）
 * ---------------------------------------------------------------------------
 */
__device__ float blockReduceSum(float val, float* shared)
{
    const unsigned int lane = threadIdx.x & 0x1f;      // 0–31
    const unsigned int wid  = threadIdx.x >> 5;        // warp ID

    // ── warp 内求和 ──
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // 每个 warp 的第一个线程把结果写到 shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // ── 第一个 warp 继续对所有 warp 的部分和做归约 ──
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.f;
    if (wid == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/*
 * ---------------------------------------------------------------------------
 * Argmin kernel : 对输入张量在指定维度执行 argmin
 * 仅支持连续 (contiguous) Tensor。
 * 维度拆分:
 *    input shape  : [outer_size, reduce_size, inner_size]
 *    output shape : [outer_size, inner_size]
 * 每个线程处理一个 (outer_idx, inner_idx) 对，应在 reduce_size 上顺序扫描。
 * ---------------------------------------------------------------------------
 */
template <typename scalar_t>
__global__ void argmin_dim_kernel(const scalar_t* __restrict__ input,
                                  int64_t* __restrict__ output,
                                  int64_t outer_size,
                                  int64_t reduce_size,
                                  int64_t inner_size,
                                  int64_t total_threads)
{
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    // 反算坐标
    const int64_t outer_idx = tid / inner_size;
    const int64_t inner_idx = tid % inner_size;

    // 该 (outer_idx, inner_idx) 对应到 input 中的起始位置
    int64_t base_offset = (outer_idx * reduce_size * inner_size) + inner_idx;

    scalar_t min_val  = input[base_offset]; // r == 0
    int64_t  min_idx  = 0;

    // 顺序扫描 reduce 维
    for (int64_t r = 1; r < reduce_size; ++r) {
        scalar_t val = input[base_offset + r * inner_size];
        if (val < min_val) {
            min_val = val;
            min_idx = r;
        }
    }

    output[tid] = min_idx; // 写入 argmin 结果
}

/*
 * ---------------------------------------------------------------------------
 * C++ 包装函数
 * ---------------------------------------------------------------------------
 */
torch::Tensor kb_1_52_Argmin_over_a_dimension_wrapper(torch::Tensor arg0,
                                                      int64_t      dim_in)
{
    TORCH_CHECK(arg0.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(arg0.is_contiguous(),
                "Only contiguous tensors are supported for this op");

    // 处理负 dim
    int64_t dim = dim_in;
    if (dim < 0) dim += arg0.dim();
    TORCH_CHECK(dim >= 0 && dim < arg0.dim(),
                "Reduction dim is out of bounds");

    // 计算 outer / reduce / inner size
    const auto sizes = arg0.sizes();
    int64_t outer_size  = 1;
    int64_t inner_size  = 1;
    int64_t reduce_size = sizes[dim];

    for (int64_t i = 0; i < dim; ++i) outer_size *= sizes[i];
    for (int64_t i = dim + 1; i < arg0.dim(); ++i) inner_size *= sizes[i];

    // 输出尺寸 = 输入尺寸移除 reduce 维
    std::vector<int64_t> out_sizes;
    out_sizes.reserve(arg0.dim() - 1);
    for (int64_t i = 0; i < arg0.dim(); ++i)
        if (i != dim) out_sizes.push_back(sizes[i]);

    torch::Tensor output = torch::empty(
        out_sizes,
        torch::TensorOptions()
            .dtype(torch::kInt64)
            .device(arg0.device()));

    // 调度 kernel
    const int64_t total_threads = outer_size * inner_size;
    const int threads = 256;
    const int blocks  = (total_threads + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(arg0.scalar_type(), "argmin_dim_kernel", ([&] {
        argmin_dim_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                arg0.data_ptr<scalar_t>(),
                output.data_ptr<int64_t>(),
                outer_size,
                reduce_size,
                inner_size,
                total_threads);
    }));

    // CUDA 内核错误检查（可选）
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ",
                cudaGetErrorString(err));

    return output;
}