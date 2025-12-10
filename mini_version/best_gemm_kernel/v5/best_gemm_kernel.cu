#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// ------------------------------------------------------------------
// KERNEL: gemm_kernel 
// ------------------------------------------------------------------
__global__ void gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    // 处理四个垂直相邻的输出tile（行方向coarsening为4）
    // Asub的四个平面用于缓存(row0,row1,row2,row3)对应的A子块
    // Bsub用于缓存当前tile的B子块
    __shared__ float Asub[4][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    // 每个线程块覆盖四个连续的行tile：tile_y, tile_y+1, tile_y+2, tile_y+3
    const int baseTileY = blockIdx.y * 4;
    const int baseRow0 = baseTileY * BLOCK_SIZE;
    const int baseRow1 = baseRow0 + BLOCK_SIZE;
    const int baseRow2 = baseRow1 + BLOCK_SIZE;
    const int baseRow3 = baseRow2 + BLOCK_SIZE;

    // 对于完全越界的块，直接返回
    if (baseRow0 >= N) {
        return;
    }

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // 每个线程负责 (row0, col) 到 (row3, col)
    const int row0 = baseRow0 + ty;
    const int row1 = baseRow1 + ty;
    const int row2 = baseRow2 + ty;
    const int row3 = baseRow3 + ty;
    const int col  = blockIdx.x * blockDim.x + tx;

    const int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float sum1 = 0.0f; // 累加 (row0, col)
    float sum2 = 0.0f; // 累加 (row1, col)
    float sum3 = 0.0f; // 累加 (row2, col)
    float sum4 = 0.0f; // 累加 (row3, col)

    // 行是否在有效范围内（块内一致的判断）
    const bool hasRow1 = (baseRow1 < N);
    const bool hasRow2 = (baseRow2 < N);
    const bool hasRow3 = (baseRow3 < N);

    // 预加载第0个tile到共享内存
    if (numTiles > 0) {
        int kA0 = 0 * BLOCK_SIZE + tx; // A的列索引
        int kB0 = 0 * BLOCK_SIZE + ty; // B的行索引

        // 加载当前tile到共享内存；越界时填0
        Asub[0][ty][tx] = (row0 < N && kA0 < N) ? A[row0 * N + kA0] : 0.0f;
        Asub[1][ty][tx] = (hasRow1 && row1 < N && kA0 < N) ? A[row1 * N + kA0] : 0.0f;
        Asub[2][ty][tx] = (hasRow2 && row2 < N && kA0 < N) ? A[row2 * N + kA0] : 0.0f;
        Asub[3][ty][tx] = (hasRow3 && row3 < N && kA0 < N) ? A[row3 * N + kA0] : 0.0f;

        Bsub[ty][tx] = (kB0 < N && col < N) ? B[kB0 * N + col] : 0.0f;
    }

    __syncthreads(); // 确保第0个tile加载完毕

    // 主循环：利用寄存器做“下一tile”的预取，实现流水化
    for (int tile = 0; tile < numTiles; ++tile) {
        // 预取下一tile的数据到寄存器（避免额外共享内存占用）
        float nextA0 = 0.0f;
        float nextA1 = 0.0f;
        float nextA2 = 0.0f;
        float nextA3 = 0.0f;
        float nextB  = 0.0f;

        const int nextTile = tile + 1;
        if (nextTile < numTiles) {
            const int kA_next = nextTile * BLOCK_SIZE + tx;
            const int kB_next = nextTile * BLOCK_SIZE + ty;

            if (row0 < N && kA_next < N) {
                nextA0 = A[row0 * N + kA_next];
            }
            if (hasRow1 && row1 < N && kA_next < N) {
                nextA1 = A[row1 * N + kA_next];
            }
            if (hasRow2 && row2 < N && kA_next < N) {
                nextA2 = A[row2 * N + kA_next];
            }
            if (hasRow3 && row3 < N && kA_next < N) {
                nextA3 = A[row3 * N + kA_next];
            }
            if (kB_next < N && col < N) {
                nextB = B[kB_next * N + col];
            }
        }

        // 计算阶段：使用当前共享内存tile
        // 优化：在每个四步迭代开始时预取4个连续的B值与各行对应的4个A值，
        // 并在行间交错执行FMA，最大化指令级并行与FMA利用率。
        #pragma unroll
        for (int kBase = 0; kBase < BLOCK_SIZE; kBase += 4) {
            // 预取4个连续的B
            float b0 = Bsub[kBase + 0][tx];
            float b1 = Bsub[kBase + 1][tx];
            float b2 = Bsub[kBase + 2][tx];
            float b3 = Bsub[kBase + 3][tx];

            // 预取每一行对应的4个连续A
            float a00 = Asub[0][ty][kBase + 0];
            float a01 = Asub[0][ty][kBase + 1];
            float a02 = Asub[0][ty][kBase + 2];
            float a03 = Asub[0][ty][kBase + 3];

            float a10 = Asub[1][ty][kBase + 0];
            float a11 = Asub[1][ty][kBase + 1];
            float a12 = Asub[1][ty][kBase + 2];
            float a13 = Asub[1][ty][kBase + 3];

            float a20 = Asub[2][ty][kBase + 0];
            float a21 = Asub[2][ty][kBase + 1];
            float a22 = Asub[2][ty][kBase + 2];
            float a23 = Asub[2][ty][kBase + 3];

            float a30 = Asub[3][ty][kBase + 0];
            float a31 = Asub[3][ty][kBase + 1];
            float a32 = Asub[3][ty][kBase + 2];
            float a33 = Asub[3][ty][kBase + 3];

            // 交错FMA：先用b0更新四行，再用b1、b2、b3
            sum1 = fmaf(a00, b0, sum1);
            sum2 = fmaf(a10, b0, sum2);
            sum3 = fmaf(a20, b0, sum3);
            sum4 = fmaf(a30, b0, sum4);

            sum1 = fmaf(a01, b1, sum1);
            sum2 = fmaf(a11, b1, sum2);
            sum3 = fmaf(a21, b1, sum3);
            sum4 = fmaf(a31, b1, sum4);

            sum1 = fmaf(a02, b2, sum1);
            sum2 = fmaf(a12, b2, sum2);
            sum3 = fmaf(a22, b2, sum3);
            sum4 = fmaf(a32, b2, sum4);

            sum1 = fmaf(a03, b3, sum1);
            sum2 = fmaf(a13, b3, sum2);
            sum3 = fmaf(a23, b3, sum3);
            sum4 = fmaf(a33, b3, sum4);
        }

        // 结束当前tile计算后，同步，随后把寄存器中预取的数据写入共享内存
        __syncthreads();

        if (nextTile < numTiles) {
            // 写入下一tile到共享内存
            Asub[0][ty][tx] = nextA0;
            Asub[1][ty][tx] = hasRow1 ? nextA1 : 0.0f;
            Asub[2][ty][tx] = hasRow2 ? nextA2 : 0.0f;
            Asub[3][ty][tx] = hasRow3 ? nextA3 : 0.0f;
            Bsub[ty][tx]    = nextB;
        }

        __syncthreads(); // 确保下一tile加载完成后再进入下一轮计算
    }

    // 写回结果，带边界检查
    if (row0 < N && col < N) {
        C[row0 * N + col] = sum1;
    }
    if (hasRow1 && row1 < N && col < N) {
        C[row1 * N + col] = sum2;
    }
    if (hasRow2 && row2 < N && col < N) {
        C[row2 * N + col] = sum3;
    }
    if (hasRow3 && row3 < N && col < N) {
        C[row3 * N + col] = sum4;
    }
}

// ------------------------------------------------------------------
// WRAPPER: gemm_cuda (这是PyTorch和CUDA之间的桥梁)
// ------------------------------------------------------------------
torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    
    // --- 输入验证 ---
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions mismatch");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(M == N && K == N, "This naive example assumes square N=M=K matrices");
    auto C = torch::zeros({M, N}, A.options());

    // --- 内核启动配置 ---
    const int block_dim_x = BLOCK_SIZE;
    const int block_dim_y = BLOCK_SIZE;
    const int grid_dim_x = (N + block_dim_x - 1) / block_dim_x;
    const int grid_dim_y = (N + block_dim_y - 1) / block_dim_y;
    dim3 blocks(grid_dim_x, grid_dim_y);
    dim3 threads(block_dim_x, block_dim_y);

    // --- 启动内核 ---
    gemm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    // --- 错误检查 ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error in gemm_kernel: " + std::string(cudaGetErrorString(err)));
    }
    return C;
}