__global__ void sharedABMultiply(float *a, float* b, float *c, int N, int K)
{
    /*The shares memory can only be claimed by one 1D array*/
    extern __shared__ float tile[]; // Declare a shared memory 
    /*So in order to divide it into two, we assign one array to the start of the 
    original array and another one to the middle*/
    float* aTile = tile; // Partition for matrix a
    int tile_dim = blockDim.x; // Tile dimension
    float* bTile = &tile[tile_dim * tile_dim]; // Partition for matrix

    int row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the column index

    float sum = 0.0f; // Initialize sum to 0

    int index = threadIdx.y * tile_dim + threadIdx.x; // Calculate the index for the 1D array

    // Calculate the number of tiles (R = K/W)
    int R = K / tile_dim;

    for (int i = 0; i < R; i++) // Loop over the tiles of the matrices
    {
        aTile[index] = a[row * K + i * tile_dim + threadIdx.x]; // Copy the elements of matrix a to the shared memory
        bTile[index] = b[(i * tile_dim + threadIdx.y) * N + col]; // Copy the elements of matrix b to the shared memory

        __syncthreads(); // Synchronize the threads: threads need data that they themselves have not written to shared memory

        for (int j = 0; j < tile_dim; j++) // Loop over the dimension of the matrices
        {
            sum += aTile[threadIdx.y * tile_dim + j] * bTile[j * tile_dim + threadIdx.x]; // Multiply elements of matrices a and b and add to sum
            //printf("aTile[%d] = %f, bTile[%d] = %f\n", threadIdx.y * tile_dim + j, aTile[threadIdx.y * tile_dim + j], j * tile_dim + threadIdx.x, bTile[j * tile_dim + threadIdx.x]);
        }

        __syncthreads(); /* Synchronize the threads: If this were not there, 
        some threads might start loading data for the next tile before all computations
         for the current tile are complete, leading to incorrect results.*/
    }

    c[row*N+col] = sum; // Store the result in matrix c
}