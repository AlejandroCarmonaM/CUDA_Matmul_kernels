__global__ void sharedABMultiply(float *a, float* b, float *c, int N)
{
    /*The shares memory can only be claimed by one 1D array*/
    extern __shared__ float tile[]; // Declare a shared memory 
    /*So in order to divide it into two, we assign one array to the start of the 
    original array and another one to the middle*/
    int tile_dim = blockDim.x; // Get the tile dimension
    float* aTile = tile; // Partition for matrix a
    float* bTile = &tile[tile_dim * tile_dim]; // Partition for matrix

    int row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the column index

    float sum = 0.0f; // Initialize sum to 0

    aTile[threadIdx.y*tile_dim + threadIdx.x] = a[row*tile_dim+threadIdx.x]; // Copy the elements of matrix a to the shared memory
    bTile[threadIdx.y*tile_dim + threadIdx.x] = b[threadIdx.y*N+col]; // Copy the elements of matrix b to the shared memory
    __syncthreads();
    //warp usa datos de B leídos por otro warp del bloque
    for (int i = 0; i < tile_dim; i++) {
        sum += aTile[threadIdx.y*tile_dim + i] * bTile[i*tile_dim + threadIdx.x]; // Multiply the elements of matrix a with the elements of matrix b
    }
    c[row*N+col] = sum; // Store the result in matrix c
}