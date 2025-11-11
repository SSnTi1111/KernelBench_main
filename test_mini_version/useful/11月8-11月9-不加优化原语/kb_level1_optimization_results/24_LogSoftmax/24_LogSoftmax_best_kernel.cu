#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_24_LogSoftmax_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

// [重要] CUDA 辅助归约函数先于 kernel 定义

// Warp 内归约 - 求和
__device__ inline float warpReduceSum(float val) {
    unsigned mask = 0xFFFFFFFFu;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// Warp 内归约 - 最大值
__device__ inline float warpReduceMax(float val) {
    unsigned mask = 0xFFFFFFFFu;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

// Block 级归约 - 求和
__device__ float blockReduceSum(float val, float* shm) {
    int lane = threadIdx.x & (warpSize - 1);
    int wid  = threadIdx.x / warpSize;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;

    // 每个 warp 内归约
    val = warpReduceSum(val);

    // 每个 warp 的 lane 0 写入共享内存
    if (lane == 0) {
        shm[wid] = val;
    }
    __syncthreads();

    // 第一个 warp 进行最终归约
    float blockVal = 0.0f;
    if (wid == 0) {
        blockVal = (lane < numWarps) ? shm[lane] : 0.0f;
        blockVal = warpReduceSum(blockVal);
        if (lane == 0) {
            shm[0] = blockVal;
        }
    }
    __syncthreads();
    return shm[0];
}

// Block 级归约 - 最大值
__device__ float blockReduceMax(float val, float* shm) {
    int lane = threadIdx.x & (warpSize - 1);
    int wid  = threadIdx.x / warpSize;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;

    // 每个 warp 内归约
    val = warpReduceMax(val);

    // 每个 warp 的 lane 0 写入共享内存
    if (lane == 0) {
        shm[wid] = val;
    }
    __syncthreads();

    // 第一个 warp 进行最终归约
    float blockVal = -INFINITY;
    if (wid == 0) {
        blockVal = (lane < numWarps) ? shm[lane] : -INFINITY;
        blockVal = warpReduceMax(blockVal);
        if (lane == 0) {
            shm[0] = blockVal;
        }
    }
    __syncthreads();
    return shm[0];
}

// CUDA 内核实现：沿着 dim=1（列）进行 LogSoftmax
__global__ void logsoftmax_rowwise_kernel(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int rows,
                                          int cols) {
    // 线程粗化因子：每个线程一次性处理多个相邻元素
    constexpr int elements_per_thread = 4;

    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_x = x + static_cast<size_t>(row) * cols;
    float* row_y = y + static_cast<size_t>(row) * cols;

    // 动态共享内存仅用于归约缓冲（每个 warp 一个槽位）
    extern __shared__ float redbuf[];

    // 共享内存行缓存（仅在列数 <= 上限时启用，或用于大行的分块缓存）
    constexpr int SMEM_ROW_CAP = 4096;
    __shared__ float s_row[SMEM_ROW_CAP];

    // Warp-level tiling 参数
    const int lane     = threadIdx.x & (warpSize - 1);
    const int warp_id  = threadIdx.x / warpSize;
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    const int warp_tile_elems = warpSize * elements_per_thread; // 32*4 = 128

    if (cols <= SMEM_ROW_CAP) {
        // 快速路径：整行缓存到共享内存，采用 warp 级平铺加载，保证合并访问与高空间局部性
        for (int tile_base = warp_id * warp_tile_elems;
             tile_base < cols;
             tile_base += warp_tile_elems * num_warps) {
#pragma unroll
            for (int k = 0; k < elements_per_thread; ++k) {
                int idx = tile_base + k * warpSize + lane;
                if (idx < cols) {
#if __CUDA_ARCH__ >= 700
                    s_row[idx] = __ldg(row_x + idx);
#else
                    s_row[idx] = row_x[idx];
#endif
                }
            }
        }
        __syncthreads();

        // 第一步：计算行最大值 (数值稳定) - 使用 warp 平铺访问共享内存
        float local_max = -INFINITY;
        for (int tile_base = warp_id * warp_tile_elems;
             tile_base < cols;
             tile_base += warp_tile_elems * num_warps) {
#pragma unroll
            for (int k = 0; k < elements_per_thread; ++k) {
                int idx = tile_base + k * warpSize + lane;
                if (idx < cols) {
                    float v = s_row[idx];
                    local_max = fmaxf(local_max, v);
                }
            }
        }
        float row_max = blockReduceMax(local_max, redbuf);

        // 第二步：计算 sum(exp(x - max))
        float local_sum = 0.0f;
        for (int tile_base = warp_id * warp_tile_elems;
             tile_base < cols;
             tile_base += warp_tile_elems * num_warps) {
#pragma unroll
            for (int k = 0; k < elements_per_thread; ++k) {
                int idx = tile_base + k * warpSize + lane;
                if (idx < cols) {
                    float val = s_row[idx] - row_max;
                    local_sum += __expf(val);
                }
            }
        }
        float sum_exp = blockReduceSum(local_sum, redbuf);

        // 计算 logsumexp
        float lse = logf(sum_exp) + row_max;

        // 第三步：写出 y = x - logsumexp
        for (int tile_base = warp_id * warp_tile_elems;
             tile_base < cols;
             tile_base += warp_tile_elems * num_warps) {
#pragma unroll
            for (int k = 0; k < elements_per_thread; ++k) {
                int idx = tile_base + k * warpSize + lane;
                if (idx < cols) {
                    row_y[idx] = s_row[idx] - lse;
                }
            }
        }
    } else {
        // 大行路径：单次分块遍历，加载一次分块后在共享内存上同时完成 tile_max 和 tile_sum 的计算，
        // 并用在线 log-sum-exp 合并策略跨 tile 累积全局最大值与和。
        const int tile_size = SMEM_ROW_CAP;

        __shared__ float run_max;
        __shared__ float run_sum;
        if (threadIdx.x == 0) {
            run_max = -INFINITY;
            run_sum = 0.0f;
        }
        __syncthreads();

        for (int row_tile_base = 0; row_tile_base < cols; row_tile_base += tile_size) {
            int tile_cols = cols - row_tile_base;
            if (tile_cols > tile_size) tile_cols = tile_size;

            // 将当前分块加载到共享内存（合并访问）
            for (int tile_base = warp_id * warp_tile_elems;
                 tile_base < tile_cols;
                 tile_base += warp_tile_elems * num_warps) {
#pragma unroll
                for (int k = 0; k < elements_per_thread; ++k) {
                    int idx = tile_base + k * warpSize + lane;
                    if (idx < tile_cols) {
#if __CUDA_ARCH__ >= 700
                        s_row[idx] = __ldg(row_x + row_tile_base + idx);
#else
                        s_row[idx] = row_x[row_tile_base + idx];
#endif
                    }
                }
            }
            __syncthreads();

            // 计算该分块的最大值
            float local_max = -INFINITY;
            for (int tile_base = warp_id * warp_tile_elems;
                 tile_base < tile_cols;
                 tile_base += warp_tile_elems * num_warps) {
#pragma unroll
                for (int k = 0; k < elements_per_thread; ++k) {
                    int idx = tile_base + k * warpSize + lane;
                    if (idx < tile_cols) {
                        float v = s_row[idx];
                        local_max = fmaxf(local_max, v);
                    }
                }
            }
            float tile_max = blockReduceMax(local_max, redbuf); // 已同步

            // 基于分块最大值计算该分块的 sum(exp(x - tile_max))
            float local_sum = 0.0f;
            for (int tile_base = warp_id * warp_tile_elems;
                 tile_base < tile_cols;
                 tile_base += warp_tile_elems * num_warps) {
#pragma unroll
                for (int k = 0; k < elements_per_thread; ++k) {
                    int idx = tile_base + k * warpSize + lane;
                    if (idx < tile_cols) {
                        local_sum += __expf(s_row[idx] - tile_max);
                    }
                }
            }
            float tile_sum = blockReduceSum(local_sum, redbuf); // 已同步

            // 在线 log-sum-exp 合并：更新全局 run_max 与 run_sum
            if (threadIdx.x == 0) {
                float new_max = fmaxf(run_max, tile_max);
                float scaled_prev = (run_sum == 0.0f) ? 0.0f : run_sum * __expf(run_max - new_max);
                float scaled_tile = tile_sum * __expf(tile_max - new_max);
                run_sum = scaled_prev + scaled_tile;
                run_max = new_max;
            }
            __syncthreads();
            // 本分块处理完成，继续到下一分块（无需再次加载）
        }

        // 计算 logsumexp
        float lse = logf(run_sum) + run_max;

        // 最终写回：直接从全局内存流式读取并写出
        for (int tile_base = warp_id * warp_tile_elems;
             tile_base < cols;
             tile_base += warp_tile_elems * num_warps) {
#pragma unroll
            for (int k = 0; k < elements_per_thread; ++k) {
                int idx = tile_base + k * warpSize + lane;
                if (idx < cols) {
#if __CUDA_ARCH__ >= 700
                    float vx = __ldg(row_x + idx);
#else
                    float vx = row_x[idx];
#endif
                    row_y[idx] = vx - lse;
                }
            }
        }
    }
}

// C++ Wrapper 实现
torch::Tensor kb_24_LogSoftmax_wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "Input tensor must be on CUDA device.");
    TORCH_CHECK(arg0.dtype() == torch::kFloat32, "Input tensor must be float32.");
    TORCH_CHECK(arg0.dim() == 2, "Input tensor must be 2D (batch_size, dim).");
    auto input = arg0.contiguous();

    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));

    auto output = torch::empty_like(input);

    const int threads = 256; // 保持为 warpSize 的倍数
    const int blocks = rows;
    const int warps = (threads + 31) / 32;
    const size_t shmem_bytes = warps * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();
    logsoftmax_rowwise_kernel<<<blocks, threads, shmem_bytes, stream.stream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "logsoftmax_rowwise_kernel launch failed: ", cudaGetErrorString(err));

    return output;
}