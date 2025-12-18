import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

////////////////////////////////////////////////////////////////
//                      CUDA KERNEL                          //
////////////////////////////////////////////////////////////////
template<typename T>
__device__ __forceinline__ T sigmoid_device(T x);

template<>
__device__ __forceinline__ float sigmoid_device<float>(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

template<typename T>
__global__ void sigmoid_kernel(const T* __restrict__ input,
                               T* __restrict__ output,
                               const int64_t numel) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = sigmoid_device<T>(input[i]);
    }
}

////////////////////////////////////////////////////////////////
//                HOST WRAPPER FUNCTION                       //
////////////////////////////////////////////////////////////////
torch::Tensor sigmoid_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Float,
                "Only float32 tensors are currently supported");

    auto output = torch::empty_like(input);

    const int64_t numel   = input.numel();
    const int     threads = 256;
    const int     blocks  = (numel + threads - 1) / threads;

    // Launch kernel
    sigmoid_kernel<float><<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );

    return output;
}
'''

cpp_src = r'''
torch::Tensor sigmoid_forward(torch::Tensor input);
'''

sigmoid_cuda = load_inline(
    name='sigmoid_cuda',
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=['sigmoid_forward'],
    with_cuda=True,
    verbose=True,                         # keep compilation log
    extra_cuda_cflags=['-O3', '--ptxas-options=-v'],  # show PTXAS info
)

class ModelNew(nn.Module):
    """
    CUDA-accelerated model that performs Sigmoid activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid_mod = sigmoid_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid_mod.sigmoid_forward(x)