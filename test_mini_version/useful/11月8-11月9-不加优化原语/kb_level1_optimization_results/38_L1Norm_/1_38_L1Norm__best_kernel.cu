#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_38_L1Norm__wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
// PyTorch 2.1+ 移除了 c10::cuda::getCurrentCUDAStream
// 使用 at::cuda::getCurrentCUDAStream() 代替
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>  // 添加 CUDAGuard 头文件

// [CUDA 辅助函数定义 - 在 kernel 之前定义]

// Warp 内归约
__device__ __forceinline__ float warpReduceSum(float val) {
    // 全掩码
    unsigned mask = 0xFFFFFFFFu;
    // 逐步对半规约
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// Block 级别归约到单个值（返回值仅在第一个 warp 的线程中有效，特别是 threadIdx.x == 0 上）
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x & (warpSize - 1); // 线程在 warp 内的索引
    int wid  = threadIdx.x >> 5;             // warp 索引

    // 先进行 warp 内归约
    val = warpReduceSum(val);

    // 每个 warp 的 lane 0 把部分和写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 只有第一个 warp 继续对各 warp 的结果做最终归约
    float sum = 0.0f;
    int warpCount = (blockDim.x + warpSize - 1) / warpSize;
    if (wid == 0) {
        sum = (lane < warpCount) ? shared[lane] : 0.0f;
        sum = warpReduceSum(sum);
    }
    return sum;
}

// CUDA 内核：对输入按 dim=1 执行 L1 归一化：out[i, j] = x[i, j] / mean(abs(x[i, :]))
__global__ void l1norm_dim1_kernel(const float* __restrict__ x,
                                   float* __restrict__ out,
                                   int64_t rows,
                                   int64_t cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[]; // 大小需为 warp 数量

    // 计算该行 |x| 的和
    float thread_sum = 0.0f;
    const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(cols);

    // 优化：每个线程优先处理两个元素，实现更高的内存合并访问。
    // 当列数恰好为 2 * blockDim.x 时，采用无循环的直接索引路径；
    // 否则，采用步长为 2 * blockDim.x 的双元素处理循环，确保通用性与正确性。
    int64_t base = static_cast<int64_t>(threadIdx.x) * 2;
    int64_t step = static_cast<int64_t>(blockDim.x) * 2;

    if (cols == step) {
        // 直接索引路径：每线程处理两个元素
        if (base < cols) {
            float v0 = x[row_offset + static_cast<size_t>(base)];
            thread_sum += fabsf(v0);
        }
        if (base + 1 < cols) {
            float v1 = x[row_offset + static_cast<size_t>(base + 1)];
            thread_sum += fabsf(v1);
        }
    } else {
        // 通用路径：每线程以步长 2*blockDim.x 处理两个元素
        for (int64_t col = base; col < cols; col += step) {
            float v0 = x[row_offset + static_cast<size_t>(col)];
            thread_sum += fabsf(v0);
            int64_t col1 = col + 1;
            if (col1 < cols) {
                float v1 = x[row_offset + static_cast<size_t>(col1)];
                thread_sum += fabsf(v1);
            }
        }
    }

    // 归约到一个 block 和
    float total_sum = blockReduceSum(thread_sum, sdata);

    // 计算 mean(|x|)
    if (threadIdx.x == 0) {
        sdata[0] = total_sum / static_cast<float>(cols);
    }
    __syncthreads();

    float mean_abs = sdata[0];

    // 写出归一化结果（同样采用双元素处理，保持一致的访问模式）
    if (cols == step) {
        if (base < cols) {
            size_t idx0 = row_offset + static_cast<size_t>(base);
            out[idx0] = x[idx0] / mean_abs; // 与 PyTorch 行为一致，不加 epsilon
        }
        if (base + 1 < cols) {
            size_t idx1 = row_offset + static_cast<size_t>(base + 1);
            out[idx1] = x[idx1] / mean_abs;
        }
    } else {
        for (int64_t col = base; col < cols; col += step) {
            size_t idx0 = row_offset + static_cast<size_t>(col);
            out[idx0] = x[idx0] / mean_abs;
            int64_t col1 = col + 1;
            if (col1 < cols) {
                size_t idx1 = row_offset + static_cast<size_t>(col1);
                out[idx1] = x[idx1] / mean_abs;
            }
        }
    }
}

// C++ Wrapper 实现
torch::Tensor kb_1_38_L1Norm__wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "kb_1_38_L1Norm__wrapper: input must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "kb_1_38_L1Norm__wrapper: input must be float32");
    TORCH_CHECK(arg0.dim() == 2, "kb_1_38_L1Norm__wrapper: only 2D tensors are supported (N x D)");

    c10::cuda::OptionalCUDAGuard device_guard(arg0.device());  // 修复：使用正确的命名空间

    // 确保内存连续
    auto x = arg0.contiguous();

    const int64_t rows = x.size(0);
    const int64_t cols = x.size(1);

    // 分配输出
    auto out = at::empty_like(x);

    // 配置 kernel
    const int threads = 256; // 每个 block 线程数
    const int blocks  = static_cast<int>(rows); // 每行一个 block
    const int num_warps = (threads + 31) / 32;
    const size_t shmem_bytes = static_cast<size_t>(num_warps) * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();

    // 启动 kernel
    l1norm_dim1_kernel<<<blocks, threads, shmem_bytes, stream.stream()>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    // 检查 kernel 启动错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kb_1_38_L1Norm__wrapper: kernel launch failed with error: ",
                cudaGetErrorString(err));

    return out;
}