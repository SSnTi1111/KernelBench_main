#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_42_Max_Pooling_2D_wrapper(torch::Tensor arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <cfloat>
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

// CUDA 内核实现: NCHW 布局的一般化 MaxPool2D
__global__ void maxpool2d_nchw_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel, int stride, int padding, int dilation
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (tid >= total) return;

    int ow = tid % outW;
    int tmp = tid / outW;
    int oh = tmp % outH;
    tmp /= outH;
    int c = tmp % C;
    int n = tmp / C;

    int start_h = oh * stride - padding;
    int start_w = ow * stride - padding;

    float maxval = -FLT_MAX;

    for (int kh = 0; kh < kernel; ++kh) {
        int ih = start_h + kh * dilation;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kernel; ++kw) {
            int iw = start_w + kw * dilation;
            if (iw < 0 || iw >= W) continue;
            int in_idx = ((n * C + c) * H + ih) * W + iw;
            float v = input[in_idx];
            if (v > maxval) maxval = v;
        }
    }

    int out_idx = ((n * C + c) * outH + oh) * outW + ow;
    output[out_idx] = maxval;
}

// C++ Wrapper 实现
torch::Tensor kb_1_42_Max_Pooling_2D_wrapper(torch::Tensor arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4) {
    TORCH_CHECK(arg0.is_cuda(), "arg0 must be a CUDA tensor");
    TORCH_CHECK(arg0.dtype() == torch::kFloat32, "arg0 must be float32");
    TORCH_CHECK(arg0.dim() == 4, "arg0 must be 4D (N, C, H, W)");

    // 参数: kernel_size, stride, padding, dilation
    int64_t kernel = arg1;
    int64_t stride = arg2;
    int64_t padding = arg3;
    int64_t dilation = arg4;

    TORCH_CHECK(kernel > 0, "kernel_size must be > 0");
    TORCH_CHECK(stride > 0, "stride must be > 0");
    TORCH_CHECK(dilation > 0, "dilation must be > 0");
    TORCH_CHECK(padding >= 0, "padding must be >= 0");

    auto x = arg0.contiguous();

    int64_t N64 = x.size(0);
    int64_t C64 = x.size(1);
    int64_t H64 = x.size(2);
    int64_t W64 = x.size(3);

    int64_t effective_kernel = dilation * (kernel - 1) + 1;
    int64_t outH64 = (H64 + 2 * padding - effective_kernel) / stride + 1;
    int64_t outW64 = (W64 + 2 * padding - effective_kernel) / stride + 1;
    if (outH64 < 0) outH64 = 0;
    if (outW64 < 0) outW64 = 0;

    torch::Tensor y = torch::empty({N64, C64, outH64, outW64}, x.options());

    int total = static_cast<int>(N64 * C64 * outH64 * outW64);
    if (total == 0) {
        return y;
    }

    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    int N = static_cast<int>(N64);
    int C = static_cast<int>(C64);
    int H = static_cast<int>(H64);
    int W = static_cast<int>(W64);
    int outH = static_cast<int>(outH64);
    int outW = static_cast<int>(outW64);
    int ksz = static_cast<int>(kernel);
    int str = static_cast<int>(stride);
    int pad = static_cast<int>(padding);
    int dil = static_cast<int>(dilation);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    maxpool2d_nchw_kernel<<<blocks, threads, 0, stream>>>(
        x_ptr, y_ptr,
        N, C, H, W,
        outH, outW,
        ksz, str, pad, dil
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "maxpool2d_nchw_kernel launch failed: ", cudaGetErrorString(err));

    return y;
}