import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA source (kernels + C++/ATen host wrappers)
# ---------------------------------------------------------------------------
source = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid_func(scalar_t x) {
    return scalar_t(1) / (scalar_t(1) + exp(x));
}

/* ---------------------------------------------------------
 * Scalar fallback kernel : one-element per thread
 * ------------------------------------------------------- */
template <typename scalar_t>
__global__ void sigmoid_kernel_scalar(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      const int64_t numel) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = sigmoid_func(input[idx]);
    }
}

/* ---------------------------------------------------------
 * Vectorised kernel : VEC elements per thread
 * VEC = 4 for float (float4, 16-byte transaction)
 *     = 2 for double (double2, 16-byte transaction)
 * The last (numel % VEC) elements are processed by a
 * single thread (vec_idx == 0) inside the same kernel.
 * ------------------------------------------------------- */
template <typename scalar_t , int VEC>
__global__ void sigmoid_kernel_vec(const scalar_t* __restrict__ input,
                                   scalar_t*       __restrict__ output,
                                   const int64_t   vec_elems,
                                   const int64_t   tail_start,
                                   const int64_t   tail_size) {
    using VecT = typename std::conditional< (sizeof(scalar_t)==4),
                                            float4,              // 4 x fp32 = 16 B
                                            double2               // 2 x fp64 = 16 B
                                          >::type;

    const int64_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* ---------------- Aligned, vectorised path ---------------- */
    if (vec_idx < vec_elems) {
        VecT v = reinterpret_cast<const VecT*>(input)[vec_idx];

        scalar_t* v_elem = reinterpret_cast<scalar_t*>(&v);
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            v_elem[i] = sigmoid_func(v_elem[i]);
        }

        reinterpret_cast<VecT*>(output)[vec_idx] = v;
    }

    /* ---------------- Tail handling by one thread ------------- */
    if (tail_size && vec_idx == 0) {
        for (int64_t j = 0; j < tail_size; ++j) {
            const int64_t idx = tail_start + j;
            output[idx] = sigmoid_func(input[idx]);
        }
    }
}

/* ---------------------------------------------------------
 * Host launcher
 * ------------------------------------------------------- */
torch::Tensor sigmoid_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must reside on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);
    const int64_t numel = input.numel();
    const int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream();

    // Fast path : fp32 / fp64 with vectorised kernel
    if (input.scalar_type() == at::kFloat || input.scalar_type() == at::kDouble) {

        if (input.scalar_type() == at::kFloat) {
            using scalar_t = float;
            constexpr int  VEC = 4;
            const int64_t  vec_elems  = numel / VEC;
            const int64_t  tail_start = vec_elems * VEC;
            const int64_t  tail_sz    = numel - tail_start;
            const int64_t  blocks     = (vec_elems + threads - 1) / threads;

            if (blocks > 0) {
                sigmoid_kernel_vec<scalar_t, VEC><<<blocks, threads, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    vec_elems,
                    tail_start,
                    tail_sz);
            } else if (tail_sz) {
                // Fallback to scalar kernel if vector part is empty
                const int64_t blocks_tail = (tail_sz + threads - 1) / threads;
                sigmoid_kernel_scalar<scalar_t><<<blocks_tail, threads, 0, stream>>>(
                    input.data_ptr<scalar_t>() + tail_start,
                    output.data_ptr<scalar_t>() + tail_start,
                    tail_sz);
            }
        } else { // double
            using scalar_t = double;
            constexpr int  VEC = 2;
            const int64_t  vec_elems  = numel / VEC;
            const int64_t  tail_start = vec_elems * VEC;
            const int64_t  tail_sz    = numel - tail_start;
            const int64_t  blocks     = (vec_elems + threads - 1) / threads;

            if (blocks > 0) {
                sigmoid_kernel_vec<scalar_t, VEC><<<blocks, threads, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    vec_elems,
                    tail_start,
                    tail_sz);
            } else if (tail_sz) {
                const int64_t blocks_tail = (tail_sz + threads - 1) / threads;
                sigmoid_kernel_scalar<scalar_t><<<blocks_tail, threads, 0, stream>>>(
                    input.data_ptr<scalar_t>() + tail_start,
                    output.data_ptr<scalar_t>() + tail_start,
                    tail_sz);
            }
        }

    } else {
        /* Generic scalar kernel for remaining dtypes (half, bfloat16, etc.) */
        const int64_t blocks = (numel + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),
                                            "sigmoid_forward_cuda_scalar", ([&] {
            sigmoid_kernel_scalar<scalar_t><<<blocks, threads, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                numel);
        }));
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sigmoid_kernel launch failed with error code ", err);

    return output;
}
'''

# ---------------------------------------------------------------------------
# C++ function prototypes
# ---------------------------------------------------------------------------
cpp_src = r'''
torch::Tensor sigmoid_forward(torch::Tensor input);
'''

# ---------------------------------------------------------------------------
# Build & load extension
# ---------------------------------------------------------------------------
sigmoid_module = load_inline(
    name         = 'sigmoid_cuda_opt_1765933647412',
    cpp_sources  = cpp_src,
    cuda_sources = source,
    functions    = ['sigmoid_forward'],
    with_cuda    = True,
    verbose      = True,
    extra_cuda_cflags=['-O3', '--ptxas-options=-v']
)

# ---------------------------------------------------------------------------
# PyTorch Module wrapper
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    CUDA-accelerated model that applies element-wise Sigmoid.
    Mirrors the original Model interface.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid = sigmoid_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid.sigmoid_forward(x)