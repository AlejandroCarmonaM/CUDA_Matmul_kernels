/*USAGE: compara_kernels -N=<dim_mat> -W=<dim_bloq> -K=<kernel>*/
/*KERNEL VALUES:*/
#define SIMPLY_MULTIPLY 1
#define COALESCED_MULTIPLY 2
#define SHARED_AB_MULTIPLY 3
#define DEBUG

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

// includes, project
 #include <cuda.h>
 #include <cuda_runtime.h>

// These are CUDA Helper functions for initialization and error checking
#include <helper_functions.h>
#include <helper_cuda.h>
#include <timer.h>

#include "utils.h"

////////////////////////////////////////////////////////////////////////////////
// kernel includes
/*kernels: simpleMultiply, sharedABMultiply, coalescedMultiply
common header for all kernels: (float *a, float* b, float *c,int N)*/
#include "./kernels/simpleMultiply_kernel.cu"
#include "./kernels/sharedABMultiply_kernel.cu"
#include "./kernels/coalescedMultiply_kernel.cu"
////////////////////////////////////////////////////////////////////////////////

/**************REQUIREMENTS**********************/
/*Threads per Block: WxW (the number of threads per block will be the leading dimension of the tile)*/
/*Tile size: WxW*/
/*N is a multiple of W so N=WxT with T as int*/
/*Grid size: TxT*/
/*Each thread calculates one and only one element of C using a row and column of b (kernels already built like that)*/
/*Each thread block calculates the elements from a tile by multiplying a tile from A with a tile from B*/


//kernel_runner function to run kernel to specific user specs and record time in start and stop
float kernel_run_chrono(int N, int kernel, dim3 grid, dim3 block, float *A_d, float *B_d, float *C_d, int tile_dim)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    switch (kernel){
        case SIMPLY_MULTIPLY:
            //shared memory size set to simpleMultiply tile: 0
            simpleMultiply<<<grid, block>>>(A_d, B_d, C_d, N);
            break;
        case COALESCED_MULTIPLY:
            //shared memory size set to coalescedMultiply tile: a_tile_dims * sizeof(float) -> tile_dim * tile_dim * sizeof(float)
            coalescedMultiply<<<grid, block, tile_dim * tile_dim * sizeof(float) >>>(A_d, B_d, C_d, N);
            break;
        case SHARED_AB_MULTIPLY:
            //shared memory size set to sharedABMultiply tile: 2 * a_tile_dims * sizeof(float) -> 2 * tile_dim * tile_dim * sizeof(float)
            sharedABMultiply<<<grid, block, 2 * tile_dim * tile_dim * sizeof(float)>>>(A_d, B_d, C_d, N);
            break;
        default:
            printf("Invalid kernel\n");
            return 1;
    }

    // 6.2. Stop GPU timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTimeGPU;
    cudaEventElapsedTime(&elapsedTimeGPU, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTimeGPU;

}

//run_program function to run program to specific user specs
int run_test(int N, int W, int kernel)
{
    int T = N/W;
    dim3 grid(T, T); //Grid size: TxT
    dim3 block(W, W); //Threads per Block: WxW
    int tile_dim = W; //Tile size: WxW

    /*2. Allocate memory and initialiaze for matrices A, B and C*/
    //initialize_matrices(int M, int K, int N, float** A, float** B, float** C) 
    float *A, *B, *C;
    initialize_matrices(N, W, N, &A, &B, &C);

    /*4. Allocate memory for matrices A_d, B_d and C_d in the device*/
    float *A_d, *B_d, *C_d;
    //allocate_device_matrices(int M_gpu, int K, int N, float** A_d, float** B_d, float** C_d)
    allocate_device_matrices(N, W, N, &A_d, &B_d, &C_d);

    /*5. Copy matrices A and B to the device*/
    // matrix_device_copy(float *A, float *B, float *A_d, float *B_d, int N, int W)
    matrix_device_copy(A, B, A_d, B_d, N, W);

    /*6. Call the kernel that the user has chosen*/
    
    // 6.1. Start GPU timer
    //kernel run chrono: run kernel to specific user specs leaving the result in C_d and record time in start and stop
    float elapsedTimeGPU = kernel_run_chrono(N, kernel, grid, block, A_d, B_d, C_d, tile_dim);

    /*7. Copy matrix C from the device to the host*/
    checkCudaErrors(cudaMemcpy(C, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    /*8. Print the execution time of the kernel*/
    //8.1 print Header
    printf("Kernel, N, W, Time\n");
    //8.2 print values
    printf("%d, %d, %d, %f\n", kernel, N, W, elapsedTimeGPU);
    

    /*EXTRA: Print all Matrices */
    printf("Matrix A:\n");
    print_matrix(A, N, W);
    printf("\n");
    printf("Matrix B:\n");
    print_matrix(B, W, N);
    printf("\n");
    printf("Matrix C:\n");
    print_matrix(C, N, N);
    printf("\n");

    //Check up results: int matrix_check(float *A, float *B, float *C, int N, int W, int kernel)
    #ifdef DEBUG
    int result = matrix_check(A, B, C, N, W, kernel);
    if(result == 0)
    {
        printf("RESULT OK: N=%d, W=%d, K=%d\n", N, W, kernel);
    }
    else
    {
        printf("RESULT FAILED: N=%d, W=%d, K=%d\n", N, W, kernel);
    }
    #endif

    /*9. Free memory*/
    free(A);
    free(B);
    free(C);
    checkCudaErrors(cudaFree(A_d));
    checkCudaErrors(cudaFree(B_d));
    checkCudaErrors(cudaFree(C_d));

    return 0;
}

//main function
int main(int argc, char **argv){
    /*1. parse arguments: compara_kernels -N=<dim_mat> -W=<dim_bloq> -K=<kernel>*/ 
    int N, W, kernel;
    if (argc != 4){
        printf("Usage: compara_kernels -N=<dim_mat> -W=<dim_bloq> -K=<kernel>\n");
        printf("KERNEL VALUES:\n");
        printf("1: SIMPLY_MULTIPLY\n");
        printf("2: COALESCED_MULTIPLY\n");
        printf("3: SHARED_AB_MULTIPLY\n");
        return 1;
    }
    sscanf(argv[1], "-N=%d", &N);
    sscanf(argv[2], "-W=%d", &W);
    sscanf(argv[3], "-K=%d", &kernel);
    if (N % W != 0){
        printf("N must be a multiple of W\n");
        return 1;
    }
    run_test(N, W, kernel);
    
}