/*USAGE: compara_kernels -N=<dim_mat> -W=<dim_bloq> -K=<kernel>*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#define DEBUG


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
#include "sharedABMultiply_kernel_hyper_tile.cu"

////////////////////////////////////////////////////////////////////////////////


/**************REQUIREMENTS**********************/
/*C_MxN=A_MxK B_KxN*/
/*M, N, K multiple of W */
/*Grid size: SxT (S=M/W, T=N/W)*/
/*Threads per Block: WxW (the number of threads per block will be the leading dimension of the tile)*/
/*Tile size: WxW*/
/*R=K/W*/

//sharedABMultiply<<<grid, block, 2 * tile_dim * tile_dim * sizeof(float)>>>(A_d, B_d, C_d, N, K);
float kernel_run_chrono(int N, int K, dim3 grid, dim3 block, float *A_d, float *B_d, float *C_d, int tile_dim)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    sharedABMultiply<<<grid, block, 2 * tile_dim * tile_dim * sizeof(float)>>>(A_d, B_d, C_d, N, K);

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
int run_test(int M, int N, int K, int W)
{
    //1. Set the dimmensions of the grid and block
    int T = N/W;
    int S = M/W;
    dim3 grid(T, S); //Grid size: SxT
    dim3 block(W, W); //Threads per Block: WxW
    int tile_dim = W; //Tile size: WxW

    /*2. Allocate memory and initialiaze for matrices A, B and C*/
    //initialize_matrices(int M, int K, int N, float** A, float** B, float** C) 
    float *A, *B, *C;
    initialize_matrices(M, K, N, &A, &B, &C);

    /*4. Allocate memory for matrices A_d, B_d and C_d in the device*/
    float *A_d, *B_d, *C_d;
    //allocate_device_matrices(int M_gpu, int K, int N, float** A_d, float** B_d, float** C_d)
    allocate_device_matrices(M, K, N, &A_d, &B_d, &C_d);

    /*5. Copy matrices A and B to the device*/
    //matrix_device_copy_mnk(float *A, float *B, float *A_d, float *B_d, int M, int N, int K)
    matrix_device_copy_mnk(A, B, A_d, B_d, M, N, K);

    /*6. Call the kernel hypertile*/
    //kernel_run_chrono(int N, int K, dim3 grid, dim3 block, float *A_d, float *B_d, float *C_d, int tile_dim)
    float elapsedTimeGPU = kernel_run_chrono(N, K, grid, block, A_d, B_d, C_d, tile_dim);

    /*7. Copy matrix C from the device to the host*/
    checkCudaErrors(cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    //8.1 print Header
    printf("M=%d, N=%d, K=%d, W=%d\n", M, N, K, W);
    printf("GPU Kernel Time: %f\n", elapsedTimeGPU);



    //print matrices A, B and C
    printf("Matrix A\n");
    print_matrix(A, M, K);
    printf("Matrix B\n");
    print_matrix(B, K, N);
    printf("Matrix C\n");
    print_matrix(C, M, N);

    /*9. Check the result using a CPU program*/
    #ifdef DEBUG
    {
        int result = matrix_check_v3(A, B, C, M, N, K, W);
        if(result == 0)
        {
            printf("TEST PASSED: M=%d, N=%d, K=%d, W=%d\n", M, N, K, W);
        }
        else
        {
            printf("TEST FAILED: M=%d, N=%d, K=%d, W=%d\n", M, N, K, W);
        }
    }
    #endif

    /*10. Free memory*/
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
     /*1. parse arguments: mulmat_1G -M=<dim_mat> -N=<dim_mat> -K=<dim_mat> -W=<dim_bloq>*/
    int M, N, K, W;
    if (argc != 5){
        printf("Usage: mulmat_1G -M=<dim_mat> -N=<dim_mat> -K=<dim_mat> -W=<dim_bloq>\n");
        return 0;
    }
    //Read the arguments
    for (int i = 1; i < argc; i++){
        if (sscanf(argv[i], "-M=%d", &M) == 1){
            continue;
        }
        if (sscanf(argv[i], "-N=%d", &N) == 1){
            continue;
        }
        if (sscanf(argv[i], "-K=%d", &K) == 1){
            continue;
        }
        if (sscanf(argv[i], "-W=%d", &W) == 1){
            continue;
        }
    }
    //Check if M, N and K are multiples of W
    if (M % W != 0 || N % W != 0 || K % W != 0){
        printf("M, N and K must be multiples of W\n");
        return 0;
    }
    if (W > 32){
        printf("W must be less than 32\n");
        return 1;
    }

    //run the test
    run_test(M, N, K, W);
    return 0;

    
}