#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_26_GELU__wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
// PyTorch 2.1+ 移除了 c10::cuda::getCurrentCUDAStream
// 使用 at::cuda::getCurrentCUDAStream() 代替
#include <ATen/cuda/CUDAContext.h>

// [重要] 在此放置所有 CUDA 辅助函数 (例如 blockReduceSum)
// (确保它们在使用它们的 kernel 之前被定义)
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Warp内归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 每个warp的第一个线程将结果写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 第一个warp进行最终归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// CUDA 内核实现: 精确 GELU (使用误差函数) - 向量化float4加载/存储
// 说明: 这里通过在内核内部将 float* 重新解释为 float4* 来实现向量化访问。
// 假设输入/输出张量是16字节对齐的（PyTorch 对 CUDA 张量通常满足该条件）。
__global__ void gelu_kernel(const float* __restrict__ x,
                            float* __restrict__ y,
                            long long N) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    const float INV_SQRT2 = 0.70710678118654752440f; // 1/sqrt(2)

    // 向量化部分：每个线程处理 float4 包
    const long long numVec = N / 4; // 可完整处理的 float4 数量
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y);

    // 手动展开：每次迭代处理 2 个 float4（i, i + stride）
    for (long long i = tid; i < numVec; i += stride * 2) {
        long long i0 = i;
        long long i1 = i + stride;

        // 预取2个向量（在边界内）
        float4 v4_0 = x4[i0];
        float4 v4_1;
        bool has1 = (i1 < numVec);
        if (has1) v4_1 = x4[i1];

        // 计算 erff()，为 2 个 float4 共 8 个元素（按可用性进行）
        // i0
        float r0_0 = 0.5f * v4_0.x * (1.0f + erff(v4_0.x * INV_SQRT2));
        float r0_1 = 0.5f * v4_0.y * (1.0f + erff(v4_0.y * INV_SQRT2));
        float r0_2 = 0.5f * v4_0.z * (1.0f + erff(v4_0.z * INV_SQRT2));
        float r0_3 = 0.5f * v4_0.w * (1.0f + erff(v4_0.w * INV_SQRT2));

        // i1
        float r1_0, r1_1, r1_2, r1_3;
        if (has1) {
            r1_0 = 0.5f * v4_1.x * (1.0f + erff(v4_1.x * INV_SQRT2));
            r1_1 = 0.5f * v4_1.y * (1.0f + erff(v4_1.y * INV_SQRT2));
            r1_2 = 0.5f * v4_1.z * (1.0f + erff(v4_1.z * INV_SQRT2));
            r1_3 = 0.5f * v4_1.w * (1.0f + erff(v4_1.w * INV_SQRT2));
        }

        // 顺序写回 2 个 float4（保持边界检查）
        y4[i0] = make_float4(r0_0, r0_1, r0_2, r0_3);
        if (has1) {
            y4[i1] = make_float4(r1_0, r1_1, r1_2, r1_3);
        }
    }

    // 处理尾部（当 N % 4 != 0 时，最多剩余 3 个元素）
    const long long remStart = numVec * 4;
    if (tid == 0) {
        for (long long j = remStart; j < N; ++j) {
            float v = x[j];
            float res = 0.5f * v * (1.0f + erff(v * INV_SQRT2));
            y[j] = res;
        }
    }
}

// C++ Wrapper 实现
torch::Tensor kb_1_26_GELU__wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "kb_1_26_GELU__wrapper: input must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "kb_1_26_GELU__wrapper: only float32 supported");

    // 保证连续内存
    torch::Tensor input = arg0.contiguous();

    // 分配输出张量，形状/设备/dtype 与输入一致
    torch::Tensor output = torch::empty_like(input);

    // 元素总数
    const long long N = input.numel();

    // 计算网格/块维度
    const int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    if (blocks > 65535) {
        blocks = 65535; // 使用 grid-stride 循环覆盖所有元素
    }

    // 获取当前 CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream();

    // 调用内核
    const float* x_ptr = input.data_ptr<float>();
    float* y_ptr = output.data_ptr<float>();

    gelu_kernel<<<blocks, threads, 0, stream.stream()>>>(x_ptr, y_ptr, N);

    // 检查内核错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gelu_kernel launch failed: ", cudaGetErrorString(err));

    return output;
}