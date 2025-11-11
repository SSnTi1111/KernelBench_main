#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_21_Sigmoid_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

// Sigmoid CUDA 内核实现
__global__ void sigmoid_kernel(const float4* __restrict__ input, float4* output, int64_t num_float4, int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_float4 - 1) {
        // Full float4 vector processing
        float4 vec = __ldg(&input[idx]);
        vec.x = 1.0f / (1.0f + expf(-vec.x));
        vec.y = 1.0f / (1.0f + expf(-vec.y));
        vec.z = 1.0f / (1.0f + expf(-vec.z));
        vec.w = 1.0f / (1.0f + expf(-vec.w));
        output[idx] = vec;
    } else if (idx == num_float4 - 1) {
        // Handle the last float4 which might have padding
        float4 vec = __ldg(&input[idx]);
        int64_t base_offset = idx * 4;
        
        // Process only valid elements
        if (base_offset + 0 < num_elements) {
            vec.x = 1.0f / (1.0f + expf(-vec.x));
        }
        if (base_offset + 1 < num_elements) {
            vec.y = 1.0f / (1.0f + expf(-vec.y));
        }
        if (base_offset + 2 < num_elements) {
            vec.z = 1.0f / (1.0f + expf(-vec.z));
        }
        if (base_offset + 3 < num_elements) {
            vec.w = 1.0f / (1.0f + expf(-vec.w));
        }
        output[idx] = vec;
    }
}

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_21_Sigmoid_wrapper(torch::Tensor arg0);

// C++ Wrapper 实现
torch::Tensor kb_21_Sigmoid_wrapper(torch::Tensor arg0) {
    // 验证输入
    TORCH_CHECK(arg0.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(arg0.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    
    // 分配输出张量
    auto output = torch::empty_like(arg0);
    
    // 获取张量数据指针
    const float* input_ptr = arg0.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // 计算总元素数量
    int64_t num_elements = arg0.numel();
    
    if (num_elements == 0) {
        return output;
    }
    
    // Calculate number of float4 elements (rounded up)
    int64_t num_float4 = (num_elements + 3) / 4;
    
    // 计算网格和块维度
    int threads_per_block = 256;
    int blocks_per_grid = (num_float4 + threads_per_block - 1) / threads_per_block;
    
    // 获取当前CUDA流
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // 启动内核
    sigmoid_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        reinterpret_cast<const float4*>(input_ptr),
        reinterpret_cast<float4*>(output_ptr),
        num_float4,
        num_elements
    );
    
    // 检查内核执行是否成功
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return output;
}