#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_23_Softmax_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <cfloat>
#include <vector>
// PyTorch 2.1+ 移除了 c10::cuda::getCurrentCUDAStream
// 使用 at::cuda::getCurrentCUDAStream() 代替
#include <ATen/cuda/CUDAContext.h>

// 辅助：warp 级别归约
__inline__ __device__ float warpReduceSum(float val) {
    unsigned mask = 0xFFFFFFFFu;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__inline__ __device__ float warpReduceMax(float val) {
    unsigned mask = 0xFFFFFFFFu;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(mask, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// 块级别归约（和）
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x & (warpSize - 1);
    int wid  = threadIdx.x / warpSize;
    int nWarps = (blockDim.x + warpSize - 1) / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    float res = 0.0f;
    if (wid == 0) {
        float v = (lane < nWarps) ? shared[lane] : 0.0f;
        float sum = warpReduceSum(v);
        if (lane == 0) shared[0] = sum;
    }
    __syncthreads();
    res = shared[0];
    return res;
}

// 块级别归约（最大）
__device__ float blockReduceMax(float val, float* shared) {
    int lane = threadIdx.x & (warpSize - 1);
    int wid  = threadIdx.x / warpSize;
    int nWarps = (blockDim.x + warpSize - 1) / warpSize;

    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    float res = -FLT_MAX;
    if (wid == 0) {
        float v = (lane < nWarps) ? shared[lane] : -FLT_MAX;
        float mx = warpReduceMax(v);
        if (lane == 0) shared[0] = mx;
    }
    __syncthreads();
    res = shared[0];
    return res;
}

// 每线程处理的分块大小（寄存器缓存的微小块，控制寄存器使用）
#ifndef KB23_TILE_CHUNK
#define KB23_TILE_CHUNK 4
#endif

// CUDA 内核：按行计算 softmax (dim=1) - 使用在线（single-pass）归约得到全局 max 与 sum，随后单次写回
__global__ void softmax_rowwise_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* __restrict__ row_in = x + static_cast<size_t>(row) * cols;
    float* __restrict__ row_out = y + static_cast<size_t>(row) * cols;

    extern __shared__ float sdata[]; // 大小为 nWarps 个 float（由 wrapper 配置）

    const int T = blockDim.x;
    const int tid = threadIdx.x;
    const int lane = tid & (warpSize - 1);
    const int wid  = tid / warpSize;
    const int nWarps = (T + warpSize - 1) / warpSize;

    // 第一阶段：在线合并，单次遍历整行，计算每线程的 (m_local, s_local)
    // 在线公式：
    // m' = max(m, x); s' = s * exp(m - m') + exp(x - m')
    float m_local = -FLT_MAX;
    float s_local = 0.0f;

    for (int base = 0; base < cols; base += T * KB23_TILE_CHUNK) {
        #pragma unroll
        for (int i = 0; i < KB23_TILE_CHUNK; ++i) {
            int c = base + tid + i * T;
            float v = -FLT_MAX;
            if (c < cols) {
#if __CUDA_ARCH__ >= 350
                v = __ldg(row_in + c);
#else
                v = row_in[c];
#endif
            }
            float m_new = fmaxf(m_local, v);
            float s_new = s_local * __expf(m_local - m_new) + __expf(v - m_new);
            m_local = m_new;
            s_local = s_new;
        }
    }

    // 线程束级别归约 (m_local, s_local)
    unsigned mask = 0xFFFFFFFFu;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float m_other = __shfl_down_sync(mask, m_local, offset);
        float s_other = __shfl_down_sync(mask, s_local, offset);
        float m_new = fmaxf(m_local, m_other);
        float s_new = s_local * __expf(m_local - m_new) + s_other * __expf(m_other - m_new);
        m_local = m_new;
        s_local = s_new;
    }

    // 块内跨 warp 归约：
    // 1) 先归约得到全局行最大 row_max
    if (lane == 0) {
        sdata[wid] = m_local; // 每个 warp 的最大值
    }
    __syncthreads();

    float row_max = -FLT_MAX;
    if (wid == 0) {
        float v = (lane < nWarps) ? sdata[lane] : -FLT_MAX;
        float mx = warpReduceMax(v);
        if (lane == 0) sdata[0] = mx;
    }
    __syncthreads();
    row_max = sdata[0];

    // 2) 使用 row_max 缩放各 warp 的 s_local，合并得到全局和 row_sum
    if (lane == 0) {
        float t = s_local * __expf(m_local - row_max);
        sdata[wid] = t;
    }
    __syncthreads();

    float row_sum = 0.0f;
    if (wid == 0) {
        float v = (lane < nWarps) ? sdata[lane] : 0.0f;
        float sum = warpReduceSum(v);
        if (lane == 0) sdata[0] = sum;
    }
    __syncthreads();
    row_sum = sdata[0];

    float inv_sum = (row_sum > 0.0f) ? __fdividef(1.0f, row_sum) : 0.0f;

    // 第二阶段：再次遍历整行，直接写入归一化的 softmax 结果
    for (int base = 0; base < cols; base += T * KB23_TILE_CHUNK) {
        #pragma unroll
        for (int i = 0; i < KB23_TILE_CHUNK; ++i) {
            int c = base + tid + i * T;
            if (c < cols) {
#if __CUDA_ARCH__ >= 350
                float v = __ldg(row_in + c);
#else
                float v = row_in[c];
#endif
                float e = __expf(v - row_max) * inv_sum;
                row_out[c] = e;
            }
        }
    }
}

// C++ Wrapper 实现
torch::Tensor kb_23_Softmax_wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "kb_23_Softmax_wrapper: input must be a CUDA tensor");
    TORCH_CHECK(arg0.dtype() == torch::kFloat32, "kb_23_Softmax_wrapper: only float32 is supported");
    TORCH_CHECK(arg0.dim() == 2, "kb_23_Softmax_wrapper: input must be 2D (batch_size, num_features)");

    // 保证连续内存
    auto x = arg0.contiguous();

    int rows = static_cast<int>(x.size(0));
    int cols = static_cast<int>(x.size(1));

    auto y = torch::empty_like(x);

    // 配置 kernel
    int threads = 256; // 多数 GPU 的良好默认值，需为 32 的倍数
    dim3 block(threads);
    dim3 grid(rows);
    int nWarps = (threads + 31) / 32;
    size_t shm_bytes = nWarps * sizeof(float); // 共享内存用于块归约

    // 启动 kernel
    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();
    auto stream = at::cuda::getCurrentCUDAStream();

    softmax_rowwise_kernel<<<grid, block, shm_bytes, stream>>>(
        x_ptr, y_ptr, rows, cols
    );

    // 可选：错误检查（在扩展中通常由调用方/同步处理）
    // cudaError_t err = cudaGetLastError();
    // TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed with error: ", cudaGetErrorString(err));

    return y;
}