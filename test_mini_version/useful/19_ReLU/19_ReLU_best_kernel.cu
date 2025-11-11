#include <torch/extension.h>

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_19_ReLU_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

// CUDA 内核实现 - 向量化 ReLU (每次处理4个元素)
__global__ void relu_kernel(const float* input, float* output, int64_t num_elements) {
    // 每个线程处理4个连续元素
    int64_t base_idx = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    
    // 使用 float4 进行向量化加载和存储
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    float4* output_vec = reinterpret_cast<float4*>(output);
    
    // 计算向量化元素的数量
    int64_t num_vec_elements = (num_elements + 3) / 4;
    int64_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx < num_vec_elements) {
        float4 val = input_vec[vec_idx];
        
        // 对每个分量应用 ReLU
        val.x = (val.x > 0.0f) ? val.x : 0.0f;
        val.y = (val.y > 0.0f) ? val.y : 0.0f;
        val.z = (val.z > 0.0f) ? val.z : 0.0f;
        val.w = (val.w > 0.0f) ? val.w : 0.0f;
        
        output_vec[vec_idx] = val;
    }
}

// C++ Wrapper 实现
torch::Tensor kb_19_ReLU_wrapper(torch::Tensor arg0) {
    // 验证输入
    TORCH_CHECK(arg0.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(arg0.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    
    // 分配输出张量
    auto output = torch::empty_like(arg0);
    
    // 获取张量数据指针
    const float* input_data = arg0.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    
    // 计算总元素数
    int64_t num_elements = arg0.numel();
    
    // 计算网格和块维度 - 每个线程处理4个元素
    int threads_per_block = 256;
    int64_t num_vec_elements = (num_elements + 3) / 4; // 向上取整
    int blocks_per_grid = (num_vec_elements + threads_per_block - 1) / threads_per_block;
    
    // 获取当前CUDA流
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // 调用内核
    relu_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        input_data, output_data, num_elements
    );
    
    // 检查CUDA错误
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return output;
}