#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_49_Max_reduction_over_a_dimension_wrapper(torch::Tensor arg0, int64_t arg1);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <cfloat>
// PyTorch 2.1+ 移除了 c10::cuda::getCurrentCUDAStream
// 使用 at::cuda::getCurrentCUDAStream() 代替
#include <ATen/cuda/CUDAContext.h>

// [辅助函数] 设备端最大值函数
__device__ inline float dmaxf(float a, float b) {
    return a > b ? a : b;
}

/*
 * 优化版 CUDA 内核：
 *  1. Tx = 32, Ty = 16 ⇒ tile 覆盖 512 个元素（2 KB shared memory）
 *  2. 仍保持每 block 256 线程的 launch 配置：
 *     每个线程负责加载两行（row 与 row+8）数据来填满 16 行 tile
 *  3. 其它逻辑（索引映射、归约、同步）保持一致
 */
__global__ void max_reduce_axis_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       long long outer,
                                       long long inner,
                                       long long red_len) {
    constexpr int Tx = 32;  // 列宽（threads per warp）
    constexpr int Ty = 16;  // 行数（每 tile 的 reduction 脚本步长）
    __shared__ float sTile[Tx * Ty];   // 32 × 16 = 512 × 4B = 2 KB

    const long long total = outer * inner;
    const long long global_tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    // 线程在 block 内的 2-D 坐标
    const int lane = threadIdx.x & (Tx - 1);  // 0 … 31
    const int row_base = threadIdx.x >> 5;    // 0 … 7 （真实行，负责写两行：row_base 及 row_base+8）

    // 判断该线程是否对应到一个有效的 (outer_idx, inner_idx)
    const bool valid_thread = (global_tid < total);

    // 仅当 valid_thread 时才计算真实索引
    const long long outer_idx = valid_thread ? (global_tid / inner) : 0;
    const long long inner_idx = valid_thread ? (global_tid % inner) : 0;

    // 基址：固定在该线程负责的 (outer_idx, inner_idx) 上，下标 j 沿 reduction 维移动
    const long long base_offset = outer_idx * red_len * inner + inner_idx;

    float local_max = -FLT_MAX;

    // tile 数 = ceil(red_len / Ty)
    const long long num_tiles = (red_len + Ty - 1) / Ty;

    for (long long tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // 当前线程负责的两个 reduction 索引
        const long long j0 = tile_idx * Ty + row_base;       // row_base 行
        const long long j1 = j0 + 8;                         // row_base + 8 行

        // 协同全局加载
        float v0 = -FLT_MAX;
        float v1 = -FLT_MAX;
        if (valid_thread && j0 < red_len) {
            v0 = input[base_offset + j0 * inner];
        }
        if (valid_thread && j1 < red_len) {
            v1 = input[base_offset + j1 * inner];
        }

        // 写入共享内存（column-major）：行 * Tx + lane
        sTile[row_base * Tx + lane]       = v0;
        sTile[(row_base + 8) * Tx + lane] = v1;

        __syncthreads();

        // 每个线程纵向扫描该列完成本 tile 的归约
#pragma unroll
        for (int k = 0; k < Ty; ++k) {
            float t = sTile[k * Tx + lane];
            local_max = dmaxf(local_max, t);
        }

        __syncthreads();  // 保护下一 tile 的写入
    }

    if (valid_thread) {
        output[global_tid] = local_max;
    }
}

// C++ Wrapper 实现
torch::Tensor kb_49_Max_reduction_over_a_dimension_wrapper(torch::Tensor arg0, int64_t arg1) {
    TORCH_CHECK(arg0.is_cuda(), "Input tensor must be on CUDA device.");
    TORCH_CHECK(arg0.scalar_type() == at::ScalarType::Float,
                "Input tensor must be float32.");
    TORCH_CHECK(arg0.numel() > 0, "Input tensor must have at least one element.");

    // 处理维度参数（支持负维度）
    int64_t nDim = arg0.dim();
    TORCH_CHECK(nDim >= 1, "Input tensor must have at least 1 dimension.");
    int64_t dim = arg1 >= 0 ? arg1 : (arg1 + nDim);
    TORCH_CHECK(dim >= 0 && dim < nDim, "Reduction dim out of range.");

    // 确保连续
    auto x = arg0.contiguous();

    // 计算 outer, inner, red_len
    auto sizes = x.sizes();
    long long outer = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer *= static_cast<long long>(sizes[i]);
    }
    long long red_len = static_cast<long long>(sizes[dim]);
    long long inner = 1;
    for (int64_t i = dim + 1; i < nDim; ++i) {
        inner *= static_cast<long long>(sizes[i]);
    }

    // 构造输出尺寸（移除归约维）
    std::vector<int64_t> out_sizes;
    out_sizes.reserve(static_cast<size_t>(nDim - 1));
    for (int64_t i = 0; i < nDim; ++i) {
        if (i == dim) continue;
        out_sizes.push_back(sizes[i]);
    }

    auto out = at::empty(out_sizes, x.options());

    // 计算网格/块维度
    long long total = outer * inner;
    int threads = 256;  // 保持 256 线程/块，内核内部每线程加载两行
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks == 0) {
        // 如果 total == 0（理论上不应出现），直接返回零大小的张量
        return out;
    }

    // 启动内核
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* in_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    max_reduce_axis_kernel<<<blocks, threads, 0, stream>>>(
        in_ptr, out_ptr, outer, inner, red_len
    );

    // 检查内核错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));

    return out;
}