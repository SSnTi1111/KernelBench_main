#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_39_L2Norm__wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
// PyTorch 2.1+ 移除了 c10::cuda::getCurrentCUDAStream
// 使用 at::cuda::getCurrentCUDAStream() 代替
#include <ATen/cuda/CUDAContext.h>

// CUDA 辅助函数: Warp 内归约
__device__ __forceinline__ float warpReduceSum(float val) {
    // 使用全掩码进行 shuffle 归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// CUDA 辅助函数: Block 内归约 (对所有线程返回相同结果)
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x & (warpSize - 1);   // 线程在 warp 中的索引
    int wid  = threadIdx.x / warpSize;         // 当前线程所属 warp 的编号

    // 先做 Warp 内归约
    val = warpReduceSum(val);

    // 每个 warp 的 lane 0 将部分和写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 由 warp 0 进行最终归约
    float total = 0.0f;
    if (wid == 0) {
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        float warp_sum = (lane < numWarps) ? shared[lane] : 0.0f;
        total = warpReduceSum(warp_sum);
        if (lane == 0) {
            shared[0] = total;  // 广播到共享内存
        }
    }
    __syncthreads();
    return shared[0];
}

// CUDA 内核实现: 对二维张量按 dim=1 (列维度) 做 L2 归一化
__global__ void l2norm_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shmem[]; // 大小为 numWarps 个 float，用于归约

    const int base = row * cols;

    // 读取阶段的 128-byte (float4 = 16-byte) 对齐计算（基于 x 的地址，用于向量化读取）
    unsigned long long row_addr_x = reinterpret_cast<unsigned long long>(x) + static_cast<unsigned long long>(base) * 4ULL;
    int misalignment_bytes_x = static_cast<int>(row_addr_x & 0xF); // 等价于 % 16
    int align_cols_x = 0;
    if (misalignment_bytes_x != 0) {
        align_cols_x = (16 - misalignment_bytes_x) / 4;
    }
    if (align_cols_x > cols) align_cols_x = cols;
    int vec_start_col_x = align_cols_x;
    int vec_elements_x = cols - align_cols_x;
    int num_vec_x = vec_elements_x / 4;
    int vec_cols_x = num_vec_x * 4;
    int remain_start_col_x = align_cols_x + vec_cols_x;

    // 计算当前行的平方和（线程局部累积），包含前缀标量、对齐后的向量化和尾部标量
    float sumsq = 0.0f;

    // 标量前缀（为对齐）
    for (int col = threadIdx.x; col < align_cols_x; col += blockDim.x) {
        float v = x[base + col];
        sumsq += v * v;
    }

    // 对齐的向量化累加（基于 x 的对齐）
    const float* vec_x_ptr_sum = x + base + vec_start_col_x;
    for (int vec_idx = threadIdx.x; vec_idx < num_vec_x; vec_idx += blockDim.x) {
        float4 v = ((const float4*)vec_x_ptr_sum)[vec_idx];
        sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // 标量尾部（剩余元素）
    for (int col = remain_start_col_x + threadIdx.x; col < cols; col += blockDim.x) {
        float v = x[base + col];
        sumsq += v * v;
    }

    // 归约得到整行的平方和
    float total = blockReduceSum(sumsq, shmem);

    // 计算 1 / sqrt(total)
    // 注意：当 total == 0 时，rsqrtf(0) = inf，此时 0 * inf = NaN，符合 x / 0 的浮点行为
    float inv_norm = rsqrtf(total);

    // 写出归一化后的结果
    // 为最大化写带宽，基于 y 的地址计算 128-byte 对齐边界进行向量化存储
    unsigned long long row_addr_y = reinterpret_cast<unsigned long long>(y) + static_cast<unsigned long long>(base) * 4ULL;
    int misalignment_bytes_y = static_cast<int>(row_addr_y & 0xF);
    int align_cols_y = 0;
    if (misalignment_bytes_y != 0) {
        align_cols_y = (16 - misalignment_bytes_y) / 4;
    }
    if (align_cols_y > cols) align_cols_y = cols;
    int vec_start_col_y = align_cols_y;
    int vec_elements_y = cols - align_cols_y;
    int num_vec_y = vec_elements_y / 4;
    int vec_cols_y = num_vec_y * 4;
    int remain_start_col_y = align_cols_y + vec_cols_y;

    // 标量前缀写回（确保 y 写入对齐前的数据）
    for (int col = threadIdx.x; col < vec_start_col_y; col += blockDim.x) {
        float v = x[base + col];
        y[base + col] = v * inv_norm;
    }

    // 向量化写回（基于 y 的对齐）。为避免 x 侧未对齐的 float4 读取，使用标量读取并打包写入 float4。
    float* vec_y_ptr = y + base + vec_start_col_y;
    for (int vec_idx = threadIdx.x; vec_idx < num_vec_y; vec_idx += blockDim.x) {
        int col = vec_start_col_y + (vec_idx << 2); // 4 * vec_idx
        float a = x[base + col + 0] * inv_norm;
        float b = x[base + col + 1] * inv_norm;
        float c = x[base + col + 2] * inv_norm;
        float d = x[base + col + 3] * inv_norm;
        float4 scaled = make_float4(a, b, c, d);
        ((float4*)vec_y_ptr)[vec_idx] = scaled;
    }

    // 标量尾部写回
    for (int col = remain_start_col_y + threadIdx.x; col < cols; col += blockDim.x) {
        float v = x[base + col];
        y[base + col] = v * inv_norm;
    }
}

// C++ Wrapper 实现
torch::Tensor kb_1_39_L2Norm__wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "kb_1_39_L2Norm__wrapper: input must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "kb_1_39_L2Norm__wrapper: only float32 is supported");
    TORCH_CHECK(arg0.dim() == 2, "kb_1_39_L2Norm__wrapper: expected a 2D tensor [rows, cols]");

    auto x = arg0.contiguous();
    const int rows = static_cast<int>(x.size(0));
    const int cols = static_cast<int>(x.size(1));

    auto y = torch::empty_like(x);

    const int threads = 256; // 每个 block 的线程数
    const dim3 block(threads);
    const dim3 grid(rows);

    // 共享内存需要为每个 warp 分配一个 float
    const int numWarps = (threads + 31) / 32;
    const size_t shmem_bytes = static_cast<size_t>(numWarps) * sizeof(float);

    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    auto stream = at::cuda::getCurrentCUDAStream();

    l2norm_dim1_kernel<<<grid, block, shmem_bytes, stream>>>(
        x_ptr, y_ptr, rows, cols
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kb_1_39_L2Norm__wrapper: CUDA kernel launch failed with error: ", cudaGetErrorString(err));

    return y;
}