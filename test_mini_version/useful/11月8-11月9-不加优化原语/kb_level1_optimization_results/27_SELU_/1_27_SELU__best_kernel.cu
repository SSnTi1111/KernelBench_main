#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_27_SELU__wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
// 包含正确的头文件
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

// [可选辅助函数示例] 块内归约（此内核未使用该函数，示例保留以供参考）
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

    // 第一个 warp 进行最终归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// CUDA SELU 内核实现
// y = scale * x, if x > 0
// y = scale * alpha * (exp(x) - 1), otherwise
__global__ void selu_kernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            int64_t numel,
                            const float scale,
                            const float alpha) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // 向量化处理的上界（确保不越界，按4对齐）
    int64_t vec_end = (numel / 4) * 4;

    // 仅当线程索引按4对齐时进行float4向量化处理
    if ((idx & 3LL) == 0) {
        for (int64_t i = idx; i < vec_end; i += stride * 4) {
            // i 此时保证按4对齐（以float计），从而满足float4加载需求
            const float4 x4 = reinterpret_cast<const float4*>(in)[i / 4];

            float x0 = x4.x;
            float x1 = x4.y;
            float x2 = x4.z;
            float x3 = x4.w;

            float y0 = (x0 > 0.0f) ? x0 : alpha * (expf(x0) - 1.0f);
            float y1 = (x1 > 0.0f) ? x1 : alpha * (expf(x1) - 1.0f);
            float y2 = (x2 > 0.0f) ? x2 : alpha * (expf(x2) - 1.0f);
            float y3 = (x3 > 0.0f) ? x3 : alpha * (expf(x3) - 1.0f);

            float4 y4 = make_float4(y0 * scale, y1 * scale, y2 * scale, y3 * scale);
            reinterpret_cast<float4*>(out)[i / 4] = y4;
        }
    }

    // 处理尾部不足4个元素的标量回退
    for (int64_t i = vec_end + idx; i < numel; i += stride) {
        float x = in[i];
        float y = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
        out[i] = scale * y;
    }
}

// C++ Wrapper 实现
torch::Tensor kb_1_27_SELU__wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "arg0 must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "arg0 must be float32");
    TORCH_CHECK(arg0.is_contiguous(), "arg0 must be contiguous; call .contiguous() before passing if needed");

    // 设备上下文保护 - 使用正确的命名空间
    c10::cuda::CUDAGuard device_guard(arg0.device());

    auto x = arg0;
    auto numel = x.numel();

    auto out = at::empty_like(x);

    if (numel == 0) {
        return out;
    }

    // SELU 常数
    constexpr float scale = 1.0507009873554805f;
    constexpr float alpha = 1.6732632423543772f;

    // 启动配置
    const int threads = 256;
    // 限制 blocks 以适配所有设备，同时使用 grid-stride 覆盖全体元素
    int64_t blocks64 = (numel + threads - 1) / threads;
    int blocks = static_cast<int>(blocks64 > 65535 ? 65535 : blocks64);

    // 当前 CUDA 流
    auto stream = at::cuda::getCurrentCUDAStream();

    selu_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int64_t>(numel),
        scale,
        alpha
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}