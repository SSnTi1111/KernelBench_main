#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_45_Average_Pooling_2D_wrapper(torch::Tensor arg0, int64_t arg1);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

// CUDA 内核实现: NCHW 格式的 2D 平均池化 (padding=0, stride=kernel_size)
__global__ void avg_pool2d_nchw_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out,
    int64_t K, int64_t S
) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;
    if (index >= total) return;

    // Decode linear index into n, c, h_out, w_out (W_out is fastest varying)
    int64_t w_out = index % W_out;
    int64_t tmp = index / W_out;
    int64_t h_out = tmp % H_out;
    tmp /= H_out;
    int64_t c = tmp % C;
    int64_t n = tmp / C;

    // Compute starting coordinates in input for this output window
    const int64_t h_start = h_out * S;
    const int64_t w_start = w_out * S;

    // Base pointer for (n, c) plane
    const int64_t base_nc = ((n * C + c) * H) * W;

    // Accumulate KxK window using pointer arithmetic to reduce multiplications in loop
    float sum = 0.0f;
    int64_t row_offset = base_nc + h_start * W + w_start;
    for (int64_t dh = 0; dh < K; ++dh) {
        const int64_t row = row_offset + dh * W;
        // Unroll small inner loops heuristically for better ILP when K is small
        int64_t w = 0;
        for (; w + 3 < K; w += 4) {
            sum += input[row + w + 0];
            sum += input[row + w + 1];
            sum += input[row + w + 2];
            sum += input[row + w + 3];
        }
        for (; w < K; ++w) {
            sum += input[row + w];
        }
    }

    output[index] = sum / static_cast<float>(K * K);
}

// C++ Wrapper 实现
torch::Tensor kb_1_45_Average_Pooling_2D_wrapper(torch::Tensor arg0, int64_t arg1) {
    TORCH_CHECK(arg0.is_cuda(), "arg0 must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "arg0 must be float32");
    TORCH_CHECK(arg0.dim() == 4, "arg0 must be 4D tensor in NCHW format");

    // kernel_size = arg1; stride = kernel_size; padding = 0
    int64_t K = arg1;
    TORCH_CHECK(K > 0, "kernel_size (arg1) must be > 0");
    int64_t S = K;
    int64_t P = 0;

    auto x = arg0.contiguous();
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(H + 2 * P >= K && W + 2 * P >= K, "kernel_size is larger than input spatial dimensions");

    int64_t H_out = (H + 2 * P - K) / S + 1;
    int64_t W_out = (W + 2 * P - K) / S + 1;
    TORCH_CHECK(H_out > 0 && W_out > 0, "Computed output size is non-positive; check kernel_size/stride/padding");

    auto y = torch::empty({N, C, H_out, W_out}, x.options());

    int threads = 256;
    int64_t total = N * C * H_out * W_out;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    avg_pool2d_nchw_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out,
        K, S
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "avg_pool2d_nchw_kernel launch failed: ", cudaGetErrorString(err));

    return y;
}