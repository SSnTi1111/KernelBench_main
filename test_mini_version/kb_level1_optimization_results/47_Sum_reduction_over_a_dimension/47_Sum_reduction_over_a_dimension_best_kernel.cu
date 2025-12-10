#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_47_Sum_reduction_over_a_dimension_wrapper(torch::Tensor arg0, int64_t arg1);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

/* =========================================================================================
   Optimized kernel – unroll reduction loop by a factor of 4 for better ILP
   ========================================================================================= */
constexpr int UNROLL = 4;

__global__ void sum_reduce_dim_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t outer_n,
    int64_t reduce_n,
    int64_t inner_n
) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= inner_n) {
        return;
    }

    const int64_t o = blockIdx.y;

    // Independent partial accumulators to break dependency chains
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    // Main unrolled loop
    int64_t unrolled_end = (reduce_n / UNROLL) * UNROLL;
    for (int64_t r = 0; r < unrolled_end; r += UNROLL) {
        int64_t base = ((o * reduce_n + r) * inner_n) + i;
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            float v = x[base + static_cast<int64_t>(k) * inner_n];
            if (k == 0) acc0 += v;
            if (k == 1) acc1 += v;
            if (k == 2) acc2 += v;
            if (k == 3) acc3 += v;
        }
    }

    // Handle remainder elements
    float acc_rem = 0.f;
    for (int64_t r = unrolled_end; r < reduce_n; ++r) {
        acc_rem += x[((o * reduce_n + r) * inner_n) + i];
    }

    // Final reduction of partial sums
    float acc = (acc0 + acc1) + (acc2 + acc3) + acc_rem;

    out[o * inner_n + i] = acc;
}

// C++ Wrapper 实现 (保持不变)
torch::Tensor kb_47_Sum_reduction_over_a_dimension_wrapper(torch::Tensor arg0, int64_t arg1) {
    TORCH_CHECK(arg0.is_cuda(), "arg0 must be a CUDA tensor");
    TORCH_CHECK(arg0.dtype() == torch::kFloat32, "Only float32 dtype is supported");
    TORCH_CHECK(arg0.is_contiguous(), "arg0 must be contiguous for this kernel");

    auto x = arg0.contiguous();
    int64_t ndim = x.dim();
    TORCH_CHECK(ndim >= 1, "Input must have at least 1 dimension");

    int64_t dim = arg1;
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "Reduction dim out of range");

    auto sizes = x.sizes();

    // 计算 outer, reduce, inner
    int64_t outer_n = 1;
    for (int64_t d = 0; d < dim; ++d) outer_n *= sizes[d];
    int64_t reduce_n = sizes[dim];
    int64_t inner_n = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner_n *= sizes[d];

    // 输出形状：keepdim=True
    std::vector<int64_t> out_sizes(sizes.begin(), sizes.end());
    out_sizes[dim] = 1;
    auto out = torch::empty(out_sizes, x.options());

    // 处理 reduce_n == 0 的情况：结果为 0
    if (reduce_n == 0 || outer_n == 0 || inner_n == 0) {
        out.zero_();
        return out;
    }

    // 按我们内核的布局，输出视为 [outer, 1, inner] 的连续张量。
    // 由于 PyTorch empty 默认是 contiguous，且我们直接按线性写入，这样即可。
    // 网格/块设置
    int threads = 256;
    dim3 grid((unsigned int)inner_n, (unsigned int)outer_n);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sum_reduce_dim_kernel<<<grid, threads, 0, stream>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        outer_n,
        reduce_n,
        inner_n
    );

    // 可选：错误检查
    // TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return out;
}