#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_100_HingeLoss_wrapper(torch::Tensor arg0, torch::Tensor arg1);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>

// Block reduction helper using shared memory and warp shuffle
__device__ float blockReduceSum(float val, float* shared) {
    const int warpSize = 32;
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Reduce within each warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // Write warp results to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // First warp reduces the shared memory results
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// Kernel declaration
__global__ void hinge_loss_kernel_256(const float* __restrict__ predictions, const float* __restrict__ targets, float* __restrict__ output, int64_t N, int64_t M);

// CUDA kernel to compute hinge loss: mean(clamp(1 - predictions * targets, min=0))
__global__ void hinge_loss_kernel_256(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int64_t N,
    int64_t M
) {
    extern __shared__ float shared_mem[];
    float* shared = shared_mem;

    float sum = 0.0f;
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_threads = gridDim.x * blockDim.x;

    // Grid-stride loop over all elements
    for (int64_t idx = tid; idx < N * M; idx += total_threads) {
        int64_t i = idx / M; // row index in predictions
        int64_t j = idx % M; // column index in predictions

        // Broadcast targets[j] across the row
        float pred = predictions[idx];
        float target = targets[j];
        float loss_val = fmaxf(0.0f, 1.0f - pred * target);
        sum += loss_val;
    }

    // Reduce sum within block
    sum = blockReduceSum(sum, shared);

    // Write block result to global memory (only first thread of first warp)
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// C++ Wrapper implementation
torch::Tensor kb_100_HingeLoss_wrapper(torch::Tensor arg0, torch::Tensor arg1) {
    // Input validation
    TORCH_CHECK(arg0.is_cuda(), "arg0 must be a CUDA tensor");
    TORCH_CHECK(arg1.is_cuda(), "arg1 must be a CUDA tensor");
    TORCH_CHECK(arg0.dtype() == torch::kFloat32, "arg0 must be float32");
    TORCH_CHECK(arg1.dtype() == torch::kFloat32, "arg1 must be float32");
    TORCH_CHECK(arg0.dim() == 2, "arg0 must be 2D");
    TORCH_CHECK(arg1.dim() == 1, "arg1 must be 1D");

    auto N = arg0.size(0);
    auto M = arg0.size(1);
    TORCH_CHECK(arg1.size(0) == M, "arg1 size must match arg0's second dimension");

    // Allocate output tensor (scalar on GPU)
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(arg0.device());
    torch::Tensor output = torch::zeros({}, options);

    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int max_blocks = 65535; // GPU limit
    int64_t total_elements = N * M;
    int64_t blocks_needed = (total_elements + threads_per_block - 1) / threads_per_block;
    int blocks = static_cast<int>(std::min(blocks_needed, static_cast<int64_t>(max_blocks)));

    // Shared memory for block reduction
    size_t shared_mem_size = (threads_per_block / 32) * sizeof(float);

    // Debug output for launch parameters
    printf("Launching kernel: blocks=%d, threads_per_block=%d, total_elements=%ld\n", blocks, threads_per_block, total_elements);

    // Launch kernel
    hinge_loss_kernel_256<<<blocks, threads_per_block, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
        arg0.data_ptr<float>(),
        arg1.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        M
    );

    // Error checking after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Kernel launch error: %s\n", cudaGetErrorString(err));

    // Normalize by total number of elements to get mean
    float scale = 1.0f / static_cast<float>(total_elements);
    output.mul_(scale);

    return output;
}