#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_41_Max_Pooling_1D_wrapper(torch::Tensor arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <limits>
// PyTorch 2.1+ 使用 at::cuda::getCurrentCUDAStream()
#include <ATen/cuda/CUDAContext.h>
// 修复：添加 CUDAGuard 的正确头文件
#include <c10/cuda/CUDAGuard.h>

// [可选辅助函数示例] 归约函数（本例未使用，保留模板结构）
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Warp 内归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 每个 warp 的 lane 0 写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 第一个 warp 做最终归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// CUDA 内核实现：MaxPool1D（N, C, L） -> （N, C, out_L）
// padding 为零填充（越界位置当作 0 值）
__global__ void maxpool1d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int L, int out_L,
    int kernel_size, int stride, int padding, int dilation,
    int shared_float_cap
) {
    // 使用动态共享内存，根据传入的 shared_float_cap 控制大小
    extern __shared__ float s_data[];

    long long nc_total = (long long)N * (long long)C;
    if (nc_total <= 0 || out_L <= 0) {
        return;
    }

    int dil_extent = (kernel_size - 1) * dilation;
    bool can_use_smem = (shared_float_cap > 0) && (dil_extent + 1) <= shared_float_cap && L > 0;

    for (long long nc_id = (long long)blockIdx.x; nc_id < nc_total; nc_id += (long long)gridDim.x) {
        int n = (int)(nc_id / C);
        int c = (int)(nc_id % C);
        long long base = ((long long)n * C + c) * (long long)L;

        if (can_use_smem) {
            // 基于共享内存容量与卷积跨度直接计算 tile 的输出数
            int tile_out_max = (shared_float_cap >= (dil_extent + 1))
                ? (1 + (shared_float_cap - (dil_extent + 1)) / stride)
                : 1;
            if (tile_out_max < 1) tile_out_max = 1;
            if (tile_out_max > out_L) tile_out_max = out_L;

            int num_tiles = (out_L + tile_out_max - 1) / tile_out_max;

            for (int t = 0; t < num_tiles; ++t) {
                int o_start = t * tile_out_max;
                int remaining = out_L - o_start;
                int o_count = remaining < tile_out_max ? remaining : tile_out_max;

                // 当前 tile 中输出对应的输入跨度
                int o_min = o_start;
                int o_max = o_start + o_count - 1;

                long long min_start_pos_ll = (long long)o_min * (long long)stride - (long long)padding;
                long long max_pos_ll = (long long)o_max * (long long)stride - (long long)padding + (long long)dil_extent;

                int min_start_pos = (int)min_start_pos_ll;
                int max_pos = (int)max_pos_ll;

                int load_start = min_start_pos;
                if (load_start < 0) load_start = 0;
                int load_end = max_pos + 1;
                if (load_end > L) load_end = L;

                int load_len = load_end - load_start;
                if (load_len < 0) load_len = 0;
                if (load_len > shared_float_cap) {
                    // 保底截断至动态共享内存容量
                    load_len = shared_float_cap;
                }

                // 协同加载共享内存（越界填充 0）
                // 经过上述裁剪，保证 0 <= g_pos < L
                for (int s = threadIdx.x; s < load_len; s += blockDim.x) {
                    int g_pos = load_start + s;
                    s_data[s] = x[base + g_pos];
                }
                __syncthreads();

                // 每个线程处理多个输出，覆盖 o_count
                int outputs_per_thread = (o_count + blockDim.x - 1) / blockDim.x;
                int start_oo = threadIdx.x * outputs_per_thread;
                int end_oo = start_oo + outputs_per_thread;
                if (end_oo > o_count) end_oo = o_count;

                for (int oo = start_oo; oo < end_oo; ++oo) {
                    int o = o_start + oo;
                    long long start_pos_ll2 = (long long)o * (long long)stride - (long long)padding;
                    int start_pos = (int)start_pos_ll2;

                    // 预先计算有效的 k 范围，并将初值设为 0 以隐式处理零填充
                    int k_min = 0;
                    int k_max = kernel_size;
                    if (start_pos < 0) {
                        // ceil((-start_pos)/dilation)
                        float t0 = (-start_pos) * 1.0f / (float)dilation;
                        k_min = (int)ceilf(t0);
                    }
                    if (start_pos + (kernel_size - 1) * dilation >= L) {
                        // floor((L-1-start_pos)/dilation) + 1
                        float t1 = (float)(L - 1 - start_pos) / (float)dilation;
                        k_max = (int)floorf(t1) + 1;
                    }
                    if (k_min < 0) k_min = 0;
                    if (k_max > kernel_size) k_max = kernel_size;

                    float maxv = 0.0f;
                    for (int k = k_min; k < k_max; ++k) {
                        int pos = start_pos + k * dilation;
                        int s_pos = pos - load_start;
                        float v = s_data[s_pos];
                        maxv = fmaxf(maxv, v);
                    }

                    long long out_idx = ((long long)n * (long long)C + (long long)c) * (long long)out_L + (long long)o;
                    y[out_idx] = maxv;
                }
                __syncthreads(); // 下一个 tile 前同步，避免覆盖共享内存
            }
        } else {
            // 回退路径：不使用共享内存，每个线程处理多个输出
            for (int o = threadIdx.x; o < out_L; o += blockDim.x) {
                long long start_pos_ll = (long long)o * (long long)stride - (long long)padding;
                int start_pos = (int)start_pos_ll;

                // 预先计算有效的 k 范围，并将初值设为 0 以隐式处理零填充
                int k_min = 0;
                int k_max = kernel_size;
                if (start_pos < 0) {
                    float t0 = (-start_pos) * 1.0f / (float)dilation;
                    k_min = (int)ceilf(t0);
                }
                if (start_pos + (kernel_size - 1) * dilation >= L) {
                    float t1 = (float)(L - 1 - start_pos) / (float)dilation;
                    k_max = (int)floorf(t1) + 1;
                }
                if (k_min < 0) k_min = 0;
                if (k_max > kernel_size) k_max = kernel_size;

                float maxv = 0.0f;
                for (int k = k_min; k < k_max; ++k) {
                    int pos = start_pos + k * dilation;
                    float v = x[base + pos];
                    maxv = fmaxf(maxv, v);
                }

                long long out_idx = ((long long)n * (long long)C + (long long)c) * (long long)out_L + (long long)o;
                y[out_idx] = maxv;
            }
        }
    }
}

// C++ Wrapper 实现
torch::Tensor kb_1_41_Max_Pooling_1D_wrapper(
    torch::Tensor arg0, // x: (N, C, L), float32, CUDA
    int64_t arg1,       // kernel_size
    int64_t arg2,       // stride
    int64_t arg3,       // padding
    int64_t arg4        // dilation
) {
    TORCH_CHECK(arg0.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(arg0.dim() == 3, "Input tensor must be 3D (N, C, L)");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(arg1 > 0, "kernel_size must be > 0");
    TORCH_CHECK(arg2 > 0, "stride must be > 0");
    TORCH_CHECK(arg4 > 0, "dilation must be > 0");
    TORCH_CHECK(arg3 >= 0, "padding must be >= 0");

    // 修复：使用正确的命名空间 c10::cuda::CUDAGuard
    c10::cuda::CUDAGuard device_guard(arg0.device());

    auto x = arg0.contiguous();
    int64_t N64 = x.size(0);
    int64_t C64 = x.size(1);
    int64_t L64 = x.size(2);

    int N = static_cast<int>(N64);
    int C = static_cast<int>(C64);
    int L = static_cast<int>(L64);

    int kernel_size = static_cast<int>(arg1);
    int stride      = static_cast<int>(arg2);
    int padding     = static_cast<int>(arg3);
    int dilation    = static_cast<int>(arg4);

    // 计算输出长度（与 PyTorch 一致）
    // out_L = floor((L + 2p - d*(k-1) - 1)/s + 1)，若结果为负则夹到 0
    long long numer = (long long)L + 2LL * padding - (long long)dilation * (kernel_size - 1) - 1LL;
    long long out_L_ll = 0;
    if (numer >= 0) {
        out_L_ll = numer / stride + 1LL;
    } else {
        out_L_ll = 0;
    }
    TORCH_CHECK(out_L_ll >= 0, "Computed output length is negative");
    int out_L = static_cast<int>(out_L_ll);
    TORCH_CHECK(out_L >= 0, "Output length overflow/invalid");

    auto options = x.options();
    auto y = torch::empty({N64, C64, (int64_t)out_L}, options);

    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    int threads = 256;
    // 使用 grid-stride 循环，限制 blocks 以适配设备
    long long total = (long long)N * (long long)C * (long long)out_L;
    int max_blocks = 65535; // 兼容性上限（多数设备 x 维上限至少为此）
    int blocks = (int)std::min((total + threads - 1) / threads, (long long)max_blocks);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (blocks == 0 || out_L == 0 || N == 0 || C == 0) {
        // 边界情形：直接返回空结果（已正确尺寸）
        return y;
    }

    // 计算共享内存参数与容量目标（约 16KB，利于更高并发）
    int dil_extent = (kernel_size - 1) * dilation;
    const int target_bytes = 16384; // ~16KB to target 8 blocks/SM
    int target_floats = target_bytes / sizeof(float); // 4096

    long long max_tile_possible = (dil_extent + 1 > target_floats)
        ? 1LL
        : ((long long)target_floats - (long long)dil_extent - 1LL) / (long long)stride + 1LL;
    int tile_out_max = std::min({threads, (int)std::min(max_tile_possible, (long long)out_L), out_L});
    long long needed_span = (tile_out_max >= 1)
        ? (long long)(tile_out_max - 1) * (long long)stride + (long long)dil_extent + 1LL
        : (long long)dil_extent + 1LL;
    int shared_float_cap_val = static_cast<int>(std::min(needed_span, (long long)L));
    if (dil_extent + 1 > target_floats) {
        // 即使一个窗口也放不下目标共享容量，使用全局内存回退
        shared_float_cap_val = 0;
    }
    size_t shared_bytes = (shared_float_cap_val > 0)
        ? static_cast<size_t>(shared_float_cap_val) * sizeof(float)
        : 0;

    maxpool1d_kernel<<<blocks, threads, shared_bytes, stream>>>(
        x_ptr, y_ptr,
        N, C, L, out_L,
        kernel_size, stride, padding, dilation,
        shared_float_cap_val
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}