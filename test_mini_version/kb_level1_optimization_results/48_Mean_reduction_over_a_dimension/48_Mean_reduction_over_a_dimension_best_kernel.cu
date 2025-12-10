#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_48_Mean_reduction_over_a_dimension_wrapper(torch::Tensor arg0, int64_t arg1);

#include <torch/extension.h>
#include <vector> // 如果返回多个张量
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <stdexcept>
#include <limits>
// PyTorch 2.1+ 移除了 c10::cuda::getCurrentCUDAStream
// 使用 at::cuda::getCurrentCUDAStream() 代替
#include <ATen/cuda/CUDAContext.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_48_Mean_reduction_over_a_dimension_wrapper(torch::Tensor arg0, int64_t arg1);

// ---------------------------------------------------------------------------------
// 线程块内求和归约，返回整个 block 的和
// 目前在新的 kernel 中不再使用，但保留实现以防将来复用
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    // Warp 内归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 每个 warp 的 lane 0 写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 仅第一个 warp 进行最终归约
    float out = 0.0f;
    if (wid == 0) {
        // 前 num_warps 个线程各自读取一个部分和
        int num_warps = blockDim.x / warpSize;
        out = (lane < num_warps) ? shared[lane] : 0.0f;
        // 再进行一次 warp 级归约
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            out += __shfl_down_sync(0xFFFFFFFF, out, offset);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------
// 新版 Kernel：thread-to-data mapping 经重新设计以实现完全合并访问
//   grid.x  → outer
//   grid.y  → ceil_div(inner, BLOCK_X)
//   block.x → BLOCK_X
// 每个线程独立完成一个 (outer_idx, inner_idx) 元素在 `reduce` 维度上的均值
template<int BLOCK_X>
__global__ void mean_reduce_dim_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long outer,
    long long reduce,
    long long inner
) {
    const long long outer_idx = static_cast<long long>(blockIdx.x);
    const long long inner_idx = static_cast<long long>(blockIdx.y) * BLOCK_X + threadIdx.x;

    // 边界保护
    if (outer_idx >= outer || inner_idx >= inner) {
        return;
    }

    // 指向当前 (outer_idx, inner_idx) 的首元素
    const float* src = x + (outer_idx * reduce * inner) + inner_idx;

    float sum = 0.0f;
    #pragma unroll 4
    for (long long r = 0; r < reduce; ++r) {
        sum += src[r * inner];   // 每次跨越 inner 个元素
    }

    const long long out_offset = outer_idx * inner + inner_idx;
    y[out_offset] = sum / static_cast<float>(reduce);
}

// ---------------------------------------------------------------------------------
// C++ Wrapper 实现
torch::Tensor kb_48_Mean_reduction_over_a_dimension_wrapper(torch::Tensor arg0, int64_t arg1) {
    TORCH_CHECK(arg0.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(arg0.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(arg0.numel() > 0, "Input tensor must have at least one element");

    auto x = arg0.contiguous();
    auto sizes = x.sizes();
    int64_t ndim = static_cast<int64_t>(sizes.size());
    TORCH_CHECK(ndim >= 1, "Input tensor must have at least 1 dimension");

    int64_t dim = arg1;
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "Reduction dim is out of range");

    // 计算 outer, reduce, inner
    long long outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= static_cast<long long>(sizes[i]);
    long long reduce = static_cast<long long>(sizes[dim]);
    long long inner = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= static_cast<long long>(sizes[i]);

    TORCH_CHECK(reduce > 0, "Reduction size must be > 0");

    // 输出形状：移除 dim
    std::vector<int64_t> out_sizes;
    out_sizes.reserve(ndim - 1);
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) continue;
        out_sizes.push_back(sizes[i]);
    }

    auto options = x.options();
    auto y = torch::empty(out_sizes, options);

    // 计算 launch 配置
    constexpr int BLOCK_X = 256;                     // 必须为 32 的倍数
    dim3 block(BLOCK_X);
    dim3 grid(static_cast<unsigned int>(outer),
              static_cast<unsigned int>((inner + BLOCK_X - 1) / BLOCK_X));
    TORCH_CHECK(grid.y <= 65535, "inner dimension too large for grid.y");

    auto stream = at::cuda::getCurrentCUDAStream();

    mean_reduce_dim_kernel<BLOCK_X><<<grid, block, 0, stream.stream()>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        outer,
        reduce,
        inner
    );

    // 可选：检查内核错误
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}