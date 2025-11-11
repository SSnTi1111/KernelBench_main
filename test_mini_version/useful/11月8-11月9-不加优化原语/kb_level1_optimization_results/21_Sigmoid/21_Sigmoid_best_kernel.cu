#include <torch/extension.h>
#include <vector> // 如果返回多个张量

// C++ Wrapper 函数声明 (签名)
torch::Tensor kb_21_Sigmoid_wrapper(torch::Tensor arg0);

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <vector>
// PyTorch 2.1+ 使用 at::cuda::getCurrentCUDAStream
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

// [可选辅助函数示例] 块级归约（本内核不使用，保留以示范）
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    // warp 内归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 每个 warp 的 lane 0 写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 第一个 warp 做最终归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// 数值稳定的 Sigmoid 设备函数
__device__ inline float sigmoidf_stable(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

// 对 float2 的数值稳定 Sigmoid
__device__ inline float2 sigmoid2_stable(const float2 v) {
    // v.x
    float ax = fabsf(v.x);
    float tx = expf(-ax);
    float sx = 1.0f / (1.0f + tx);
    float rx = (v.x >= 0.0f) ? sx : (1.0f - sx);
    // v.y
    float ay = fabsf(v.y);
    float ty = expf(-ay);
    float sy = 1.0f / (1.0f + ty);
    float ry = (v.y >= 0.0f) ? sy : (1.0f - sy);
    return make_float2(rx, ry);
}

// CUDA 内核实现：warp 级分块 + 向量化(float2)逐元素 Sigmoid，带边界与回退处理
// 本版实现了8路展开的双缓冲(in-place)预取与计算流水：
// - 每次迭代处理8个float2（每lane），按照4+4两个阶段
// - 使用两组float2变量集合（a0..a3 作为当前A组，b0..b3作为当前B组），并在计算时原地覆写为下一次迭代的预取数据
// - 在A组计算时预取下一迭代A组，在B组计算时预取下一迭代B组，从而实现load-compute-store的重叠
// - 加入编译器内存屏障与warp级同步以利于指令排序与可见性（在A800上兼容）
__global__ void sigmoid_kernel(const float* __restrict__ x,
                               float* __restrict__ y,
                               int64_t N) {
    if (N <= 0) return;

    // 计算可向量化的 float2 数量
    int64_t N2 = N / 2;

    // 访问对齐检查，确保 float2 安全载入/存储
    bool aligned = ((reinterpret_cast<uintptr_t>(x) % alignof(float2)) == 0) &&
                   ((reinterpret_cast<uintptr_t>(y) % alignof(float2)) == 0);

    const int warp_sz = warpSize;

    // 向量化路径：warp级分块 + 8路展开 + 双缓冲
    if (aligned && N2 > 0 && blockDim.x >= warp_sz) {
        const float2* __restrict__ x2 = reinterpret_cast<const float2*>(x);
        float2* __restrict__ y2 = reinterpret_cast<float2*>(y);

        int lane = threadIdx.x & (warp_sz - 1);
        int warp_in_block = threadIdx.x >> 5;  // 等价于 / warp_sz
        int warps_per_block = blockDim.x >> 5;

        int64_t global_warp_id = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
        int64_t total_warps = static_cast<int64_t>(gridDim.x) * warps_per_block;

        // 每个迭代每个 warp 覆盖 warp_sz 个连续的 float2 元素
        int64_t start_i2 = static_cast<int64_t>(global_warp_id) * warp_sz + lane;
        int64_t stride_i2 = static_cast<int64_t>(total_warps) * warp_sz;

        // 8 路展开
        const int UNROLL = 8;
        int64_t chunk = stride_i2 * UNROLL;

        // 双缓冲寄存器：A组(0..3)与B组(4..7)
        float2 a0, a1, a2, a3; // 当前/下次 A 组缓冲（in-place）
        float2 b0, b1, b2, b3; // 当前/下次 B 组缓冲（in-place）

        bool have_cur = false; // 是否已有当前迭代的寄存器缓冲（第一轮需要从内存装载）

        for (int64_t base = start_i2; base < N2; base += chunk) {
            // 计算当前迭代的索引
            int64_t ia0 = base;
            int64_t ia1 = ia0 + stride_i2;
            int64_t ia2 = ia1 + stride_i2;
            int64_t ia3 = ia2 + stride_i2;

            int64_t ib0 = base + (stride_i2 << 2); // 4 * stride_i2
            int64_t ib1 = ib0 + stride_i2;
            int64_t ib2 = ib1 + stride_i2;
            int64_t ib3 = ib2 + stride_i2;

            // 预载当前迭代的8个数据（仅首轮需要），后续轮次使用上一轮预取到的寄存器值
            if (!have_cur) {
                if (ia0 < N2) a0 = x2[ia0];
                if (ia1 < N2) a1 = x2[ia1];
                if (ia2 < N2) a2 = x2[ia2];
                if (ia3 < N2) a3 = x2[ia3];

                if (ib0 < N2) b0 = x2[ib0];
                if (ib1 < N2) b1 = x2[ib1];
                if (ib2 < N2) b2 = x2[ib2];
                if (ib3 < N2) b3 = x2[ib3];

                // 编译器内存屏障，帮助形成load/compute的明确相位
                asm volatile("" ::: "memory");
                __syncwarp();
            }

            int64_t next_base = base + chunk;

            // Phase A: 计算当前A组(0..3)，同时预取下一迭代A组到相同寄存器（in-place覆盖）
            // 索引：当前 ia*，下一次 nia* = next_base + {0,1,2,3}*stride_i2
            int64_t nia0 = next_base;
            int64_t nia1 = nia0 + stride_i2;
            int64_t nia2 = nia1 + stride_i2;
            int64_t nia3 = nia2 + stride_i2;

            // A0
            if (ia0 < N2) {
                y2[ia0] = sigmoid2_stable(a0);
            }
            if (nia0 < N2) {
                a0 = x2[nia0]; // 预取下一迭代A0
            }

            // A1
            if (ia1 < N2) {
                y2[ia1] = sigmoid2_stable(a1);
            }
            if (nia1 < N2) {
                a1 = x2[nia1]; // 预取下一迭代A1
            }

            // A2
            if (ia2 < N2) {
                y2[ia2] = sigmoid2_stable(a2);
            }
            if (nia2 < N2) {
                a2 = x2[nia2]; // 预取下一迭代A2
            }

            // A3
            if (ia3 < N2) {
                y2[ia3] = sigmoid2_stable(a3);
            }
            if (nia3 < N2) {
                a3 = x2[nia3]; // 预取下一迭代A3
            }

            // 相位边界：插入编译器屏障与warp同步，以利于调度
            asm volatile("" ::: "memory");
            __syncwarp();

            // Phase B: 计算当前B组(4..7)，同时预取下一迭代B组到相同寄存器（in-place覆盖）
            // 索引：当前 ib*，下一次 nib* = next_base + {4,5,6,7}*stride_i2
            int64_t nib0 = next_base + (stride_i2 << 2);
            int64_t nib1 = nib0 + stride_i2;
            int64_t nib2 = nib1 + stride_i2;
            int64_t nib3 = nib2 + stride_i2;

            // B0
            if (ib0 < N2) {
                y2[ib0] = sigmoid2_stable(b0);
            }
            if (nib0 < N2) {
                b0 = x2[nib0]; // 预取下一迭代B0
            }

            // B1
            if (ib1 < N2) {
                y2[ib1] = sigmoid2_stable(b1);
            }
            if (nib1 < N2) {
                b1 = x2[nib1]; // 预取下一迭代B1
            }

            // B2
            if (ib2 < N2) {
                y2[ib2] = sigmoid2_stable(b2);
            }
            if (nib2 < N2) {
                b2 = x2[nib2]; // 预取下一迭代B2
            }

            // B3
            if (ib3 < N2) {
                y2[ib3] = sigmoid2_stable(b3);
            }
            if (nib3 < N2) {
                b3 = x2[nib3]; // 预取下一迭代B3
            }

            // Phase 结束：形成明确相位边界，确保良好指令排序
            asm volatile("" ::: "memory");
            __syncwarp();

            // 现在 a0..a3、b0..b3 已经承载了下一迭代的数据
            have_cur = true;
        }

        // 处理奇数剩余元素（仅由一个线程负责以避免竞态）
        if ((N & 1) && (blockIdx.x == 0) && (threadIdx.x == 0)) {
            int64_t last = N - 1;
            y[last] = sigmoidf_stable(x[last]);
        }
    } else {
        // 标量路径：同样采用 warp 级分块以保持良好的缓存局部性
        if (blockDim.x >= warp_sz) {
            int lane = threadIdx.x & (warp_sz - 1);
            int warp_in_block = threadIdx.x >> 5;
            int warps_per_block = blockDim.x >> 5;

            int64_t global_warp_id = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
            int64_t total_warps = static_cast<int64_t>(gridDim.x) * warps_per_block;

            int64_t start_i = static_cast<int64_t>(global_warp_id) * warp_sz + lane;
            int64_t stride_i = static_cast<int64_t>(total_warps) * warp_sz;

            // 基础4路展开
            for (int64_t base = start_i; base < N; base += stride_i * 4) {
                #pragma unroll 4
                for (int u = 0; u < 4; ++u) {
                    int64_t idx = base + static_cast<int64_t>(u) * stride_i;
                    if (idx < N) {
                        float vx = x[idx];
                        float ax = fabsf(vx);
                        float tx = expf(-ax);
                        float sx = 1.0f / (1.0f + tx);
                        y[idx] = (vx >= 0.0f) ? sx : (1.0f - sx);
                    }
                }
            }
        } else {
            // 保底：blockDim 小于一个 warp 时使用线程级 grid-stride，并做基础4路展开
            int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            int64_t gstride = static_cast<int64_t>(blockDim.x) * gridDim.x;

            for (int64_t base = gid; base < N; base += gstride * 4) {
                #pragma unroll 4
                for (int u = 0; u < 4; ++u) {
                    int64_t idx = base + static_cast<int64_t>(u) * gstride;
                    if (idx < N) {
                        float vx = x[idx];
                        float ax = fabsf(vx);
                        float tx = expf(-ax);
                        float sx = 1.0f / (1.0f + tx);
                        y[idx] = (vx >= 0.0f) ? sx : (1.0f - sx);
                    }
                }
            }
        }
    }
}

// C++ Wrapper 实现
torch::Tensor kb_21_Sigmoid_wrapper(torch::Tensor arg0) {
    TORCH_CHECK(arg0.is_cuda(), "kb_21_Sigmoid_wrapper: input must be a CUDA tensor");
    TORCH_CHECK(arg0.scalar_type() == at::kFloat, "kb_21_Sigmoid_wrapper: only float32 is supported");

    c10::cuda::CUDAGuard device_guard(arg0.device());
    auto x = arg0.contiguous();
    auto out = at::empty_like(x);

    int64_t N = x.numel();
    if (N == 0) {
        return out;
    }

    const int threads = 256;
    const cudaDeviceProp* props = at::cuda::getCurrentDeviceProperties();
    int sm_count = props->multiProcessorCount;
    // 选择合理的 blocks 数，使用 grid-stride loop 覆盖所有元素
    int max_blocks = sm_count * 32;
    int64_t needed_blocks = (N + threads - 1) / threads;
    int blocks = static_cast<int>(std::min<int64_t>(needed_blocks, static_cast<int64_t>(max_blocks)));
    blocks = std::max(blocks, 1);

    auto stream = at::cuda::getCurrentCUDAStream();

    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = out.data_ptr<float>();

    sigmoid_kernel<<<blocks, threads, 0, stream.stream()>>>(x_ptr, y_ptr, N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sigmoid_kernel launch failed: ", cudaGetErrorString(err));

    return out;
}