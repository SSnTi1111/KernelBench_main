#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_21_8_HardSigmoid_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
// [!!! 关键 !!!] 
// PyTorch 2.1+ 移除了 c10::cuda::getCurrentCUDAStream
// 使用 at::cuda::getCurrentCUDAStream() 代替
#include <ATen/cuda/CUDAContext.h>

// [重要] 在此放置所有 CUDA 辅助函数 (例如 blockReduceSum)
// (确保它们在使用它们的 kernel 之前被定义)
__device__ float blockReduceSum(float val, float* shared) {
    // 示例 Warp 内归约
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

// CUDA 内核实现: HardSigmoid = clamp((x + 3) / 6, 0, 1)
// 等价于 clamp(x * (1/6) + 0.5, 0, 1)
__global__ void hardsigmoid_kernel(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int64_t N) {
    // 线程相关索引与步长
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t idx4 = tid * 4;
    int64_t stride4 = static_cast<int64_t>(blockDim.x) * gridDim.x * 4;

    int64_t idx = tid;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    const float inv6 = 1.0f / 6.0f;

    // 向量化指针
    const float4* __restrict__ in4 = reinterpret_cast<const float4*>(in);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(out);

    // 向量化主循环范围（保证不越界）
    int64_t NvecEnd = (N / 4) * 4;

    // 向量化处理：一次处理4个元素
    for (int64_t i4 = idx4; i4 < NvecEnd; i4 += stride4) {
        float4 v4 = in4[i4 / 4];

        // 使用 FMA 计算未裁剪值：u = v * (1/6) + 0.5
        float ux = fmaf(v4.x, inv6, 0.5f);
        float uy = fmaf(v4.y, inv6, 0.5f);
        float uz = fmaf(v4.z, inv6, 0.5f);
        float uw = fmaf(v4.w, inv6, 0.5f);

        // 分支无关裁剪到 [0, 1]
        float yx = fminf(1.0f, fmaxf(0.0f, ux));
        float yy = fminf(1.0f, fmaxf(0.0f, uy));
        float yz = fminf(1.0f, fmaxf(0.0f, uz));
        float yw = fminf(1.0f, fmaxf(0.0f, uw));

        float4 y4 = make_float4(yx, yy, yz, yw);
        out4[i4 / 4] = y4;
    }

    // 处理剩余非4倍数的尾部元素
    for (int64_t i = idx + NvecEnd; i < N; i += stride) {
        float v = in[i];
        float u = fmaf(v, inv6, 0.5f);
        float y = fminf(1.0f, fmaxf(0.0f, u));
        out[i] = y;
    }
}

// C++ Wrapper 实现
torch::Tensor kb_21_8_HardSigmoid_wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "kb_21_8_HardSigmoid_wrapper: input must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "kb_21_8_HardSigmoid_wrapper: only float32 is supported");

    auto input = arg0.contiguous();
    auto N = input.numel();

    auto output = at::empty_like(input);

    const int threads = 256;
    // 选择合理的块数，避免过大，同时使用网格步长循环覆盖所有元素
    int64_t blocks64 = (N + threads - 1) / threads;
    // 限制 gridDim.x 以获得较好的占用率与兼容性
    int blocks = static_cast<int>(std::min<int64_t>(blocks64, 65535));

    // 获取当前 CUDA 流并启动内核
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    hardsigmoid_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kb_21_8_HardSigmoid_wrapper: kernel launch failed with error: ",
                cudaGetErrorString(err));

    return output;
}