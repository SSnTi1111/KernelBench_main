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
    // Tiled matrix multiplication using shared memory with 2x2 register tiling
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE+1];  // Added padding to avoid bank conflicts
    
    // Each thread computes a 2x2 sub-block of the output
    int row = blockIdx.y * blockDim.y * 2 + threadIdx.y * 2;
    int col = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    
    // Accumulator registers for 2x2 sub-block
    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;
    
    int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // Load tile from A into shared memory - each thread loads 2 elements
        int A_row = row;
        int A_col = tile_idx * BLOCK_SIZE + threadIdx.x * 2;
        
        if (A_row < N && A_col < N) {
            Asub[threadIdx.y * 2][threadIdx.x * 2] = A[A_row * N + A_col];
            if (A_col + 1 < N) {
                Asub[threadIdx.y * 2][threadIdx.x * 2 + 1] = A[A_row * N + A_col + 1];
            } else {
                Asub[threadIdx.y * 2][threadIdx.x * 2 + 1] = 0.0f;
            }
        } else {
            Asub[threadIdx.y * 2][threadIdx.x * 2] = 0.0f;
            Asub[threadIdx.y * 2][threadIdx.x * 2 + 1] = 0.0f;
        }
        
        // Load second row of A tile
        if (A_row + 1 < N && A_col < N) {
            Asub[threadIdx.y * 2 + 1][threadIdx.x * 2] = A[(A_row + 1) * N + A_col];
            if (A_col + 1 < N) {
                Asub[threadIdx.y * 2 + 1][threadIdx.x * 2 + 1] = A[(A_row + 1) * N + A_col + 1];
            } else {
                Asub[threadIdx.y * 2 + 1][threadIdx.x * 2 + 1] = 0.0f;
            }
        } else {
            Asub[threadIdx.y * 2 + 1][threadIdx.x * 2] = 0.0f;
            Asub[threadIdx.y * 2 + 1][threadIdx.x * 2 + 1] = 0.0f;
        }
        
        // Load tile from B into shared memory with transposed indexing - each thread loads 2 elements
        int B_row = tile_idx * BLOCK_SIZE + threadIdx.y * 2;
        int B_col = col;
        
        if (B_row < N && B_col < N) {
            Bsub[threadIdx.x * 2][threadIdx.y * 2] = B[B_row * N + B_col];
            if (B_col + 1 < N) {
                Bsub[threadIdx.x * 2 + 1][threadIdx.y * 2] = B[B_row * N + B_col + 1];
            } else {
                Bsub[threadIdx.x * 2 + 1][threadIdx.y * 2] = 0.0f;
            }
        } else {
            Bsub[threadIdx.x * 2][threadIdx.y * 2] = 0.0f;
            Bsub[threadIdx.x * 2 + 1][threadIdx.y * 2] = 0.0f;
        }
        
        // Load second row of B tile
        if (B_row + 1 < N && B_col < N) {
            Bsub[threadIdx.x * 2][threadIdx.y * 2 + 1] = B[(B_row + 1) * N + B_col];
            if (B_col + 1 < N) {
                Bsub[threadIdx.x * 2 + 1][threadIdx.y * 2 + 1] = B[(B_row + 1) * N + B_col + 1];
            } else {
                Bsub[threadIdx.x * 2 + 1][threadIdx.y * 2 + 1] = 0.0f;
            }
        } else {
            Bsub[threadIdx.x * 2][threadIdx.y * 2 + 1] = 0.0f;
            Bsub[threadIdx.x * 2 + 1][threadIdx.y * 2 + 1] = 0.0f;
        }
        
        // Synchronize to ensure all tiles are loaded
        __syncthreads();
        
        // Compute partial sums for 2x2 sub-block using shared memory
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float a0 = Asub[threadIdx.y * 2][k];
            float a1 = Asub[threadIdx.y * 2 + 1][k];
            float b0 = Bsub[threadIdx.x * 2][k];
            float b1 = Bsub[threadIdx.x * 2 + 1][k];
            
            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write 2x2 sub-block results to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum00;
        if (col + 1 < N) {
            C[row * N + col + 1] = sum01;
        }
    }
    if (row + 1 < N && col < N) {
        C[(row + 1) * N + col] = sum10;
        if (col + 1 < N) {
            C[(row + 1) * N + col + 1] = sum11;
        }
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
    // Grid dimensions reduced by factor of 2 in each dimension due to 2x2 tiling
    const int block_dim_x = BLOCK_SIZE / 2;  // 16
    const int block_dim_y = BLOCK_SIZE / 2;  // 16
    const int grid_dim_x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int grid_dim_y = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
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