#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_32_HardTanh_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// [重要] 在此放置所有 CUDA 辅助函数 (例如 blockReduceSum)
// (确保它们在使用它们的 kernel 之前被定义)
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Warp 内归约
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 每个 warp 的 lane 0 写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 使用第一个 warp 完成块内归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

__device__ __forceinline__ float hardtanh_op(float x, float minv, float maxv) {
    // 使用快速设备函数实现 clamp
    x = fminf(x, maxv);
    x = fmaxf(x, minv);
    return x;
}

// CUDA 内核实现
__global__ void hardtanh_kernel_f32(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    size_t N,
                                    float minv,
                                    float maxv) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    // 使用 float4 进行向量化访问
    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);

    // 计算向量化的起始索引与步长（以元素为单位）
    size_t idx4 = idx * 4;
    size_t stride4 = stride * 4;

    // 对齐到 4 的边界（可安全进行 float4 读写的区域长度）
    size_t vec_end = (N / 4) * 4;

    // 向量化主循环：处理前面可整除 4 的部分
    for (size_t i4 = idx4; i4 < vec_end; i4 += stride4) {
        float4 v4 = in4[i4 / 4];
        float4 v_clamped;
        v_clamped.x = hardtanh_op(v4.x, minv, maxv);
        v_clamped.y = hardtanh_op(v4.y, minv, maxv);
        v_clamped.z = hardtanh_op(v4.z, minv, maxv);
        v_clamped.w = hardtanh_op(v4.w, minv, maxv);
        out4[i4 / 4] = v_clamped;
    }

    // 处理剩余的 0-3 个尾部元素（标量路径）
    for (size_t i = idx; i < N; i += stride) {
        if (i >= vec_end) {
            float v = in[i];
            out[i] = hardtanh_op(v, minv, maxv);
        }
    }
}

// C++ Wrapper 实现
torch::Tensor kb_1_32_HardTanh_wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.device().is_cuda(), "kb_1_32_HardTanh_wrapper: input must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "kb_1_32_HardTanh_wrapper: only float32 tensors are supported");

    // 设备保护，确保在输入张量所在设备上分配与执行
    c10::cuda::CUDAGuard device_guard(arg0.device());

    // 保证内存连续
    auto x = arg0.contiguous();

    // 分配输出
    auto out = torch::empty_like(x);

    const float* in_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    const size_t N = static_cast<size_t>(x.numel());
    if (N == 0) {
        return out;
    }

    // 配置 kernel 启动参数
    int threads = 256;
    // 使用网格-步长循环，限制 grid 大小以兼容性最佳
    int64_t blocks64 = (static_cast<int64_t>(N) + threads - 1) / threads;
    if (blocks64 <= 0) blocks64 = 1;
    int max_blocks = 65535; // 兼容性好的上限
    int blocks = static_cast<int>(std::min<int64_t>(blocks64, max_blocks));

    // 获取当前 CUDA 流
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const float minv = -1.0f;
    const float maxv = 1.0f;

    // 启动 kernel
    hardtanh_kernel_f32<<<blocks, threads, 0, stream>>>(in_ptr, out_ptr, N, minv, maxv);

    // 检查 kernel 启动错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kb_1_32_HardTanh_wrapper: CUDA kernel launch failed with error: ",
                cudaGetErrorString(err));

    return out;
}