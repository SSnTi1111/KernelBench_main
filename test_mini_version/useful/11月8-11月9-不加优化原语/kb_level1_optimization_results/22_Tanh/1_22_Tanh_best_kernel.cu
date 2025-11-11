#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_1_22_Tanh_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <cstdint>
// PyTorch 2.1+ 移除了 c10::cuda::getCurrentCUDAStream
// 使用 at::cuda::getCurrentCUDAStream() 代替
#include <ATen/cuda/CUDAContext.h>

// [重要] 在此放置所有 CUDA 辅助函数 (例如 blockReduceSum)
// (确保它们在使用它们的 kernel 之前被定义)
__device__ float blockReduceSum(float val, float* shared) {
    // 示例 Warp 内归约
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Warp 内归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 每个 warp 的第一个线程将结果写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 第一个 warp 进行最终归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// 可能用到的元素级操作（在 kernel 之前定义）
__device__ __forceinline__ float tanh_op(float x) {
    return tanhf(x);
}

// CUDA 内核实现：对一维展平后的数组应用 Tanh（向量化 float4 + ILP 展开版本）
__global__ void tanh_kernel(const float* __restrict__ x,
                            float* __restrict__ y,
                            int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // 向量化指针与向量元素个数
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* y_vec = reinterpret_cast<float4*>(y);
    int64_t n_vec = n / 4;

    // 使用 ILP=3 的展开：每次迭代处理三个间隔为 stride 的 float4 块
    const int UNROLL = 3;
    int64_t vec_stride = stride; // 以 float4 为单位的步长

    // 主向量化循环：以 float4 为单位处理，并进行 ILP 展开
    for (int64_t base = idx; base < n_vec; base += vec_stride * UNROLL) {
        int64_t j0 = base;
        int64_t j1 = base + vec_stride;
        int64_t j2 = base + 2 * vec_stride;

        // 提前发起三个独立的全局内存读取请求
        float4 v0 = (j0 < n_vec) ? x_vec[j0] : make_float4(0.f, 0.f, 0.f, 0.f);
        float4 v1 = (j1 < n_vec) ? x_vec[j1] : make_float4(0.f, 0.f, 0.f, 0.f);
        float4 v2 = (j2 < n_vec) ? x_vec[j2] : make_float4(0.f, 0.f, 0.f, 0.f);

        // 交错进行 tanh 计算，提升指令级并行度
        v0.x = tanh_op(v0.x);
        v1.x = tanh_op(v1.x);
        v2.x = tanh_op(v2.x);

        v0.y = tanh_op(v0.y);
        v1.y = tanh_op(v1.y);
        v2.y = tanh_op(v2.y);

        v0.z = tanh_op(v0.z);
        v1.z = tanh_op(v1.z);
        v2.z = tanh_op(v2.z);

        v0.w = tanh_op(v0.w);
        v1.w = tanh_op(v1.w);
        v2.w = tanh_op(v2.w);

        // 写回结果，带边界检查
        if (j0 < n_vec) y_vec[j0] = v0;
        if (j1 < n_vec) y_vec[j1] = v1;
        if (j2 < n_vec) y_vec[j2] = v2;
    }

    // 处理尾部不足 4 个元素的部分（标量循环）
    for (int64_t i = n_vec * 4 + idx; i < n; i += stride) {
        y[i] = tanh_op(x[i]);
    }
}

// C++ Wrapper 实现
torch::Tensor kb_1_22_Tanh_wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "kb_1_22_Tanh_wrapper: arg0 must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == torch::kFloat32,
                "kb_1_22_Tanh_wrapper: only float32 tensors are supported");

    // 设备保护，确保在输入张量所在设备上运行
    c10::DeviceGuard guard(arg0.device());

    // 保证输入连续
    auto x = arg0.contiguous();

    // 分配连续输出
    auto out = torch::empty_like(x);

    const int64_t n = x.numel();
    if (n == 0) {
        return out;
    }

    const int threads = 256;
    // 为兼容性限制 gridDim.x 到 65535，并使用 grid-stride 循环覆盖全部元素
    int64_t blocks64 = (n + threads - 1) / threads;
    int blocks = static_cast<int>(std::min<int64_t>(blocks64, 65535));

    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    tanh_kernel<<<blocks, threads, 0, stream>>>(x_ptr, out_ptr, n);

    // 检查 kernel 启动错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kb_1_22_Tanh_wrapper: CUDA kernel launch failed with error: ",
                cudaGetErrorString(err));

    return out;
}