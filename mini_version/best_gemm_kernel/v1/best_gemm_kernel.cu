#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifndef THREAD_TILE_M
#define THREAD_TILE_M 2
#endif

#ifndef THREAD_TILE_N
#define THREAD_TILE_N 2
#endif

#ifndef UNROLL_K
#define UNROLL_K 4
#endif

// ------------------------------------------------------------------
// KERNEL: gemm_kernel 
// ------------------------------------------------------------------
__global__ void gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    // Register-blocked, tiled shared-memory GEMM kernel with software pipelining.
    // Pipeline: preload tile 0 into buf0, compute on buf (cur), while prefetching next tile to registers.
    // After compute, commit prefetched tile to the other buffer (next), sync, ping-pong buffers, repeat.

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int blockRowBase = blockIdx.y * BLOCK_SIZE;
    const int blockColBase = blockIdx.x * BLOCK_SIZE;

    // Double-buffered shared memory tiles with padding to reduce bank conflicts
    __shared__ float As_buf0[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs_buf0[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float As_buf1[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs_buf1[BLOCK_SIZE][BLOCK_SIZE + 1];

    // Mapping of compute threads to 2x2 patches inside a 32x32 tile
    const int rowTile0 = ty * THREAD_TILE_M;
    const int colTile0 = tx * THREAD_TILE_N;

    const int row0 = blockRowBase + rowTile0 + 0;
    const int row1 = row0 + 1;
    const int col0 = blockColBase + colTile0 + 0;
    const int col1 = col0 + 1;

    const bool isComputeThread =
        (ty < (BLOCK_SIZE / THREAD_TILE_M)) && (tx < (BLOCK_SIZE / THREAD_TILE_N));

    // Global indices reused across iterations
    const int aRow = blockRowBase + ty;
    const int bCol = blockColBase + tx;

    // Accumulators for the 2x2 per-thread patch
    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    // --------------------------
    // Stage 0: preload first K-tile (t = 0) into buf0
    // --------------------------
    int aCol0 = 0 + tx;
    int bRow0 = 0 + ty;

    As_buf0[ty][tx] = (aRow < N && aCol0 < N) ? A[aRow * N + aCol0] : 0.0f;
    Bs_buf0[ty][tx] = (bRow0 < N && bCol < N) ? B[bRow0 * N + bCol] : 0.0f;

    __syncthreads();

    int buf = 0; // 0 => using buf0 as current, 1 => using buf1 as current

    // --------------------------
    // Main loop over K dimension with ping-pong buffering
    // --------------------------
    for (int t = 0; t < N; t += BLOCK_SIZE) {
        // Select current and next shared-memory buffers
        float (*curAs)[BLOCK_SIZE + 1]  = (buf == 0) ? As_buf0 : As_buf1;
        float (*curBs)[BLOCK_SIZE + 1]  = (buf == 0) ? Bs_buf0 : Bs_buf1;
        float (*nextAs)[BLOCK_SIZE + 1] = (buf == 0) ? As_buf1 : As_buf0;
        float (*nextBs)[BLOCK_SIZE + 1] = (buf == 0) ? Bs_buf1 : Bs_buf0;

        const int tNext = t + BLOCK_SIZE;
        const bool hasNext = (tNext < N);

        // Prefetch the next K tile from global memory into registers to overlap with compute
        float aPref = 0.0f;
        float bPref = 0.0f;
        if (hasNext) {
            const int aColNext = tNext + tx;
            const int bRowNext = tNext + ty;
            aPref = (aRow < N && aColNext < N) ? A[aRow * N + aColNext] : 0.0f;
            bPref = (bRowNext < N && bCol < N) ? B[bRowNext * N + bCol] : 0.0f;
        }

        // Compute on the current shared-memory tile with conservative K-loop unrolling
        if (isComputeThread) {
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; k += UNROLL_K) {
                #pragma unroll
                for (int u = 0; u < UNROLL_K; ++u) {
                    float a0 = curAs[rowTile0 + 0][k + u];
                    float a1 = curAs[rowTile0 + 1][k + u];
                    float b0 = curBs[k + u][colTile0 + 0];
                    float b1 = curBs[k + u][colTile0 + 1];

                    c00 = fmaf(a0, b0, c00);
                    c01 = fmaf(a0, b1, c01);
                    c10 = fmaf(a1, b0, c10);
                    c11 = fmaf(a1, b1, c11);
                }
            }
        }

        // Commit the prefetched next tile into the next shared-memory buffers
        if (hasNext) {
            // Keep boundary guards consistent; write zeros when out-of-bounds
            nextAs[ty][tx] = (aRow < N) ? aPref : 0.0f;
            nextBs[ty][tx] = (bCol < N) ? bPref : 0.0f;
        }

        // Ensure the next tile is fully committed before it is used in the next iteration
        __syncthreads();

        // Ping-pong between buffers
        buf ^= 1;
    }

    // Write the final 2x2 results from compute threads with bounds checks
    if (isComputeThread) {
        if (row0 < N && col0 < N) C[row0 * N + col0] = c00;
        if (row0 < N && col1 < N) C[row0 * N + col1] = c01;
        if (row1 < N && col0 < N) C[row1 * N + col0] = c10;
        if (row1 < N && col1 < N) C[row1 * N + col1] = c11;
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