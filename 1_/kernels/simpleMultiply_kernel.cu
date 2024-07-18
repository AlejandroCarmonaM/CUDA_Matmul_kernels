__global__ void simpleMultiply(float *a, float* b, float *c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the column index

    int tile_dim = blockDim.x; // Get the tile dimension
    float sum = 0.0f; // Initialize sum to 0

    for (int i = 0; i < tile_dim; i++) // Loop over the dimension of the matrices
    {
        sum += a[row * tile_dim + i] * b[i * N + col]; // Multiply elements of matrices a and b and add to sum
    }

    //__syncthreads(); // Synchronize the threads

    c[row*N+col] = sum; // Store the result in matrix c
}