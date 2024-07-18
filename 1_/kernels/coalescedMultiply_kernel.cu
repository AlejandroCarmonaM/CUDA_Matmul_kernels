__global__ void coalescedMultiply(float *a, float* b, float *c, int N)
{
    extern __shared__ float aTile[]; // Declare a shared memory tile for matrix a
    int tile_dim = blockDim.x; // Get the tile dimension

    int row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the column index

    float sum = 0.0f; // Initialize sum to 0

    aTile[threadIdx.y * tile_dim + threadIdx.x] = a[row * tile_dim + threadIdx.x]; // Copy the elements of matrix a to the shared memory

    __syncthreads(); // Synchronize the threads

    for (int i = 0; i < tile_dim; i++) {
        sum += aTile[threadIdx.y * tile_dim + i] * b[i * N + col]; // Multiply the elements of matrix a with the elements of matrix b
    }

    c[row*N+col] = sum; // Store the result in matrix c
}