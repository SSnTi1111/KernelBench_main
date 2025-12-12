#include <torch/extension.h>
#include <vector>

// C++  Wrapper 函数声明
torch::Tensor kb_53_Min_reduction_over_a_dimension_wrapper(torch::Tensor arg0,
                                                           int64_t       arg1);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <cfloat>
#include <vector>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

/********************************************
 *  CUDA  辅助函数
 ********************************************/
__inline__ __device__ float warpReduceMin(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float blockReduceMin(float v) {
    __shared__ float shared[32];                // 1 float / warp   (最大 1024 线程 ==> 32 个 warp)
    int  lane = threadIdx.x & 31;               // 线程在 warp 内的 index
    int  wid  = threadIdx.x >> 5;               // warp id

    v = warpReduceMin(v);                       // Warp 内最小值

    if (lane == 0)                              // 每个 warp 写入共享内存
        shared[wid] = v;
    __syncthreads();

    // 由第 0 个 warp 归约所有 warp 的最小值
    v = (threadIdx.x < blockDim.x / 32) ? shared[lane] : FLT_MAX;
    if (wid == 0)
        v = warpReduceMin(v);

    return v;                                   // 所有线程同值
}

/********************************************
 *  CUDA  Kernel
 *
 *  说明：
 *  为了与新的内存访问优化保持兼容，kernel 在运行时动态区分
 *  `inner_size == 1`（被归约维度已位于最后一维，内存连续）与
 *  `inner_size  > 1`（保持原有跨 stride 的访问方式）两种场景。
 *
 *  • 当 inner_size == 1 时：
 *      每个线程一次性加载 kChunk（默认 8）个连续 float，
 *      使得每个 warp 形成一次 256 B 对齐事务。
 *  • 当 inner_size  > 1 时：
 *      回退到原有实现，不改变调用侧（wrapper）的行为。
 *
 *  其余逻辑（block-level 规约、grid 配置等）保持不变，因此
 *  wrapper 代码可以完全复用，满足“不修改 wrapper”这一约束。
 ********************************************/
template <int kBlockSize>
__global__ void min_reduce_dim_kernel(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int   outer_size,
                                      int   reduce_size,
                                      int   inner_size) {
    constexpr int kChunk = 8;  // 每线程一次处理的 float 数

    /* 由于 gridSize 可能小于总 slice 数，这里采用 grid-stride
       循环以便 kernel 配置在不同形状张量上都能安全运行。*/
    const int total_slices = outer_size * inner_size;

    for (int slice_idx = blockIdx.x; slice_idx < total_slices; slice_idx += gridDim.x) {

        int outer_idx = slice_idx / inner_size;    // 对应于 [0, outer_size)
        int inner_idx = slice_idx % inner_size;    // 对应于 [0, inner_size)

        float local_min = FLT_MAX;

        /* ------------------  Case 1 : inner_size == 1  ------------------ *
         * 归约维度位于张量最后一维，内存连续；此时可以用连续地址
         * 访问提升带宽利用率。                                             */
        if (inner_size == 1) {
            const int base_offset = outer_idx * reduce_size;   // 简化地址计算

#if (kChunk % 4 == 0)
            /* 若基础地址 16 字节对齐，可选用 float4 载入进一步降低请求数 */
            bool addr_aligned_16 =
                (reinterpret_cast<std::uintptr_t>(in + base_offset) & 0xF) == 0;

            if (addr_aligned_16) {
                const float4* __restrict__ in4 =
                    reinterpret_cast<const float4*>(in + base_offset);

                const int reduce_size4 = reduce_size >> 2;               // /4
                const int thread_offset4 = (threadIdx.x * kChunk) >> 2;  // /4
                const int stride4 = (kBlockSize * kChunk) >> 2;          // /4

                for (int r4 = thread_offset4; r4 < reduce_size4; r4 += stride4) {
    #pragma unroll
                    for (int i = 0; i < (kChunk >> 2); ++i) {   // kChunk / 4 次 float4
                        int idx4 = r4 + i;
                        if (idx4 < reduce_size4) {
                            float4 v4 = __ldg(in4 + idx4);
                            local_min = fminf(local_min, v4.x);
                            local_min = fminf(local_min, v4.y);
                            local_min = fminf(local_min, v4.z);
                            local_min = fminf(local_min, v4.w);
                        }
                    }
                }

                /* 处理 reduce_size 不是 4 的整数倍的尾部元素 */
                const int tail_start = reduce_size4 << 2;
                for (int idx = tail_start + threadIdx.x;
                     idx < reduce_size;
                     idx += kBlockSize) {
                    float v = __ldg(in + base_offset + idx);
                    local_min = fminf(local_min, v);
                }
            } else
#endif  // (kChunk % 4 == 0)
            {
                /* ---------- 标量批量载入路径 (对齐不足或 float4 关闭) ---------- */
                int r = threadIdx.x * kChunk;
                const int stride = kBlockSize * kChunk;

                for (; r < reduce_size; r += stride) {
    #pragma unroll
                    for (int i = 0; i < kChunk; ++i) {
                        int idx = r + i;
                        if (idx < reduce_size) {
                            float v = __ldg(in + base_offset + idx);
                            local_min = fminf(local_min, v);
                        }
                    }
                }
            }
        }
        /* ------------------  Case 2 : inner_size  > 1  ------------------ *
         * 保持原有跨 stride 的读取方式，确保与旧版接口兼容。              */
        else {
            const int base_offset =
                (outer_idx * reduce_size) * inner_size + inner_idx;

            for (int r = threadIdx.x; r < reduce_size; r += kBlockSize) {
                float v = __ldg(in + base_offset + r * inner_size);
                local_min = fminf(local_min, v);
            }
        }

        // Block 内规约
        float block_min = blockReduceMin(local_min);

        // 写回结果：block 中仅由 thread 0 执行
        if (threadIdx.x == 0)
            out[slice_idx] = block_min;
    }
}

/********************************************
 *  C++  Wrapper
 ********************************************/
torch::Tensor kb_53_Min_reduction_over_a_dimension_wrapper(torch::Tensor arg0,
                                                           int64_t       arg1) {
    TORCH_CHECK(arg0.is_cuda(),  "Input tensor must be on CUDA device");
    TORCH_CHECK(arg0.scalar_type() == torch::kFloat32,
                "Only float32 tensors are supported");

    const int64_t dim   = arg1;
    const auto    sizes = arg0.sizes();
    const int64_t ndim  = sizes.size();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim is out of range");

    // 计算 outer_size / reduce_size / inner_size
    int64_t outer_size  = 1;
    int64_t inner_size  = 1;
    const int64_t reduce_size = sizes[dim];

    for (int64_t i = 0; i < dim; ++i)        outer_size *= sizes[i];
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= sizes[i];

    // 创建输出张量 (移除被归约维度)
    std::vector<int64_t> out_sizes;
    out_sizes.reserve(ndim - 1);
    for (int64_t i = 0; i < ndim; ++i)
        if (i != dim) out_sizes.push_back(sizes[i]);

    auto out = torch::empty(out_sizes, arg0.options());

    // 确保输入是连续的，以便我们使用简单的步长公式
    auto in_contig = arg0.contiguous();

    const float* in_ptr  = in_contig.data_ptr<float>();
    float*       out_ptr = out.data_ptr<float>();

    // Kernel Launch parameters
    constexpr int kBlockSize = 256;                      // 使用不同名字避免与宏冲突
    const int grid_size = static_cast<int>(outer_size * inner_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    min_reduce_dim_kernel<kBlockSize>
        <<<grid_size, kBlockSize, 0, stream>>>(in_ptr,
                                               out_ptr,
                                               static_cast<int>(outer_size),
                                               static_cast<int>(reduce_size),
                                               static_cast<int>(inner_size));

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return out;
}