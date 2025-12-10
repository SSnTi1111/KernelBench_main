#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_51_Argmax_over_a_dimension_wrapper(torch::Tensor arg0, int64_t arg1);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cfloat>     // FLT_MAX

//=======================================================================
// CUDA Kernel : 3-D tensor argmax along an arbitrary dimension (0/1/2)
// Each thread produces exactly one output element (i.e. one argmax index)
//=======================================================================
__global__ void argmax_dim3_kernel(const float* __restrict__ input,
                                   int64_t* __restrict__ output,
                                   int64_t size0, int64_t size1, int64_t size2,
                                   int      reduce_dim)
{
    /*  Shape convention
          dim-0 : size0
          dim-1 : size1
          dim-2 : size2
    */
    size_t out_elements = 0;
    if (reduce_dim == 0)
        out_elements = size1 * size2;
    else if (reduce_dim == 1)
        out_elements = size0 * size2;
    else /* reduce_dim == 2 */
        out_elements = size0 * size1;

    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_elements) return;

    float  max_val  = -FLT_MAX;
    int64_t max_idx = 0;

    if (reduce_dim == 0) {
        // --- map flat output index -> (i1, i2)
        int64_t i1 = out_idx / size2;
        int64_t i2 = out_idx % size2;

        // --- scan along dim-0
        for (int64_t r = 0; r < size0; ++r) {
            float val = input[r * size1 * size2 + i1 * size2 + i2];
            if (val > max_val) {
                max_val  = val;
                max_idx  = r;
            }
        }
    }
    else if (reduce_dim == 1) {   //  ***   default path for the provided example   ***
        // --- map flat output index -> (i0, i2)
        int64_t i0 = out_idx / size2;
        int64_t i2 = out_idx % size2;

        // --- scan along dim-1
        for (int64_t r = 0; r < size1; ++r) {
            float val = input[i0 * size1 * size2 + r * size2 + i2];
            if (val > max_val) {
                max_val  = val;
                max_idx  = r;
            }
        }
    }
    else { // reduce_dim == 2
        // --- map flat output index -> (i0, i1)
        int64_t i0 = out_idx / size1;
        int64_t i1 = out_idx % size1;

        // --- scan along dim-2
        for (int64_t r = 0; r < size2; ++r) {
            float val = input[i0 * size1 * size2 + i1 * size2 + r];
            if (val > max_val) {
                max_val  = val;
                max_idx  = r;
            }
        }
    }

    // write back
    output[out_idx] = max_idx;
}

//=======================================================================
// C++ Wrapper
//=======================================================================
torch::Tensor kb_1_51_Argmax_over_a_dimension_wrapper(torch::Tensor arg0, int64_t arg1) {
    // -------------------- sanity checks --------------------
    TORCH_CHECK(arg0.is_cuda(),   "Input tensor must reside on CUDA device");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat,
                "Input tensor must be of type float32");
    TORCH_CHECK(arg0.dim() == 3,
                "This implementation supports 3-D tensors only (got ",
                arg0.dim(), ")");

    int64_t reduce_dim = arg1;
    TORCH_CHECK(reduce_dim >= 0 && reduce_dim < 3,
                "reduce_dim must be 0, 1 or 2 (got ", reduce_dim, ")");

    // Make sure the tensor is contiguous for simple stride math
    auto input = arg0.contiguous();

    const auto sizes = input.sizes();
    int64_t size0 = sizes[0];
    int64_t size1 = sizes[1];
    int64_t size2 = sizes[2];

    // -------------------- create output tensor --------------------
    std::vector<int64_t> out_sizes;
    if (reduce_dim == 0)
        out_sizes = {size1, size2};
    else if (reduce_dim == 1)
        out_sizes = {size0, size2};
    else // reduce_dim == 2
        out_sizes = {size0, size1};

    auto output = torch::empty(out_sizes,
                               torch::dtype(torch::kInt64)
                               .device(arg0.device()));

    // -------------------- launch kernel --------------------
    const int threads = 256;
    int64_t num_out_elements = output.numel();
    const int blocks = static_cast<int>((num_out_elements + threads - 1) / threads);

    // current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    argmax_dim3_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        size0, size1, size2,
        static_cast<int>(reduce_dim)
    );

    // optional: check for launch errors (debug builds)
#ifndef NDEBUG
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
#endif

    return output;
}