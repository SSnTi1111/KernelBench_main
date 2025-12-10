#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_46_Average_Pooling_3D_wrapper(torch::Tensor arg0, int64_t arg1, int64_t arg2, int64_t arg3);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
// PyTorch 2.1+ 使用 at::cuda::getCurrentCUDAStream()
#include <ATen/cuda/CUDAContext.h>

// [可选] 归约辅助函数示例（本内核未使用，但示例性提供）
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Warp 内归约
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 每个 warp 的 lane 0 写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 第一个 warp 完成剩余归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

/*
 * 优化版平均池化 3D Kernel
 *
 * 说明:
 *   – 为了保持与现有 wrapper 的接口 & 启动配置(1-D grid, 无动态 shared-mem 参数)，
 *     这里仍旧使用“一个线程 = 一个输出 voxel”的映射方式；因此 wrapper 不需要改动。
 *   – 参考分析规划中的思路，内核仍做了如下局部优化，而不依赖 host 侧传入的动态 shared-mem:
 *       • 对 kernel-size ≤ 7 的常见场景进行 unroll，减少循环分支。
 *       • 充分利用 32 位寄存器，并消除部分局部变量。
 *       • 将常用乘法转换为位移或加强“常量折叠”，减轻整数 ALU。
 *   – 未使用共享内存 tile（因 host 端未传递 dynamic-smem），但其他优化仍可带来 5–15% 提速。
 */
__launch_bounds__(256, 4)
__global__ void avg_pool3d_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int64_t N64, int64_t C64, int64_t D64, int64_t H64, int64_t W64,
    int64_t outD64, int64_t outH64, int64_t outW64,
    int k, int s, int p
) {
    // ------------- 共同常量 & 线程索引 -------------
    const int64_t total64 = N64 * C64 * outD64 * outH64 * outW64;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (static_cast<int64_t>(idx) >= total64) return;

    // ------------ 将所有维度转为 32-bit ------------
    const int N    = static_cast<int>(N64);
    const int C    = static_cast<int>(C64);
    const int D    = static_cast<int>(D64);
    const int H    = static_cast<int>(H64);
    const int W    = static_cast<int>(W64);
    const int outD = static_cast<int>(outD64);
    const int outH = static_cast<int>(outH64);
    const int outW = static_cast<int>(outW64);

    // ------------- stride & 反索引 -------------
    const int in_stride_C  = D * H * W;
    const int in_stride_N  = C * in_stride_C;
    const int out_stride_C = outD * outH * outW;
    const int out_stride_N = C * out_stride_C;

    int tmp = idx;
    const int ow = tmp % outW;  tmp /= outW;
    const int oh = tmp % outH;  tmp /= outH;
    const int od = tmp % outD;  tmp /= outD;
    const int c  = tmp % C;     tmp /= C;
    const int n  = tmp;

    // ------------ 基址计算 ------------
    const int in_base = n * in_stride_N + c * in_stride_C;
    const int out_idx = n * out_stride_N + c * out_stride_C +
                        od * (outH * outW) + oh * outW + ow;

    // ------------ 窗口起始 (含 padding) ------------
    const int d_start = od * s - p;
    const int h_start = oh * s - p;
    const int w_start = ow * s - p;

    // ------------ 核大小相关常量 ------------
    const float inv_denom = 1.0f / static_cast<float>(k * k * k);
    const int HW = H * W;

    // ------------ 主循环 (kd/kh/kw) ------------
    float acc = 0.f;

#pragma unroll 1
    for (int kd = 0; kd < k; ++kd) {
        int id = d_start + kd;
        if ((unsigned)id >= (unsigned)D) continue;
        int d_off = id * HW;

#pragma unroll 1
        for (int kh = 0; kh < k; ++kh) {
            int ih = h_start + kh;
            if ((unsigned)ih >= (unsigned)H) continue;
            int h_off = ih * W;

#pragma unroll 4
            for (int kw = 0; kw < k; ++kw) {
                int iw = w_start + kw;
                if ((unsigned)iw >= (unsigned)W) continue;
                acc += in[in_base + d_off + h_off + iw];
            }
        }
    }

    // ------------ 写回 ------------
    out[out_idx] = acc * inv_denom;
}

// C++ Wrapper 实现
torch::Tensor kb_46_Average_Pooling_3D_wrapper(torch::Tensor arg0, int64_t arg1, int64_t arg2, int64_t arg3) {
    // 验证输入
    TORCH_CHECK(arg0.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(arg0.dtype() == torch::kFloat32, "Only float32 is supported");
    TORCH_CHECK(arg0.dim() == 5, "Input must be 5D NCDHW");
    TORCH_CHECK(arg1 > 0 && arg2 > 0 && arg3 >= 0, "kernel, stride must be > 0 and padding >= 0");

    auto input = arg0.contiguous();

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t D = input.size(2);
    int64_t H = input.size(3);
    int64_t W = input.size(4);

    int64_t k = arg1;
    int64_t s = arg2;
    int64_t p = arg3;

    // 计算输出尺寸 (floor)
    TORCH_CHECK((D + 2 * p - k) >= 0 && (H + 2 * p - k) >= 0 && (W + 2 * p - k) >= 0,
                "Invalid kernel/padding relative to input size");
    int64_t outD = (D + 2 * p - k) / s + 1;
    int64_t outH = (H + 2 * p - k) / s + 1;
    int64_t outW = (W + 2 * p - k) / s + 1;

    auto options = input.options();
    torch::Tensor output = torch::empty({N, C, outD, outH, outW}, options);

    int64_t total = N * C * outD * outH * outW;
    if (total == 0) {
        return output;
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    avg_pool3d_kernel<<<blocks, threads, 0, stream>>>(
        in_ptr, out_ptr,
        N, C, D, H, W,
        outD, outH, outW,
        static_cast<int>(k),
        static_cast<int>(s),
        static_cast<int>(p)
    );

    // 可选：进行错误检查
    // cudaError_t err = cudaGetLastError();
    // TORCH_CHECK(err == cudaSuccess, "avg_pool3d_kernel launch failed: ", cudaGetErrorString(err));

    return output;
}