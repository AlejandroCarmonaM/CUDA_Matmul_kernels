/*USAGE: compara_kernels –N=<dim_mat> -W=<dim_bloq> -K=<kernel>*/
/*KERNEL VALUES:*/
#define SIMPLY_MULTIPLY 1
#define COALESCED_MULTIPLY 2
#define SHARED_AB_MULTIPLY 3

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
    kernel_run_chrono(N, kernel, grid, block, A_d, B_d, C_d, tile_dim);

    /*7. Copy matrix C from the device to the host*/
    checkCudaErrors(cudaMemcpy(C, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    /*8. Check the result using a CPU program*/
    //int matrix_check(float *A, float *B, float *C, int N, int W, int kernel)
    int result = matrix_check(A, B, C, N, W, kernel);
    
    //if the test passed, display the configuration (N, W, kernel)
    if(result == 0)
    {
        printf("TEST PASSED: N=%d, W=%d, K=%d\n", N, W, kernel);
    }
    else
    {
        printf("TEST FAILED: N=%d, W=%d, K=%d\n", N, W, kernel);
        exit(EXIT_FAILURE);
    }

    /*9. Free memory*/
    free(A);
    free(B);
    free(C);
    checkCudaErrors(cudaFree(A_d));
    checkCudaErrors(cudaFree(B_d));
    checkCudaErrors(cudaFree(C_d));

    return 0;
}

void exhaustive_test()
{
    //We must comply with the restrictions imposed by the problem in the tests:
    //Threads per Block: WxW (the number of threads per block will be the leading dimension of the tile)
    //Tile size: WxW
    //N is a multiple of W so N=WxT with T as int
    //Grid size: TxT

    //Values:
    // N = [1, 4096]
    // W = [1, 32]
    // K = [1, 3]

    //n_max up to 4096 and w_max up to 32
    int n_max = 256;
    int w_max = 32;
    int kernel = 1;
    int kernel_passed[3] = {0, 0, 0}; //Array to store if a kernel passed all tests


    //We will test all possible (allowed by the restrictions) combinations of N, W and K
    int passed_tests = 0;
    int failed_tests = 0;

    //Loop for kernels 
    for(kernel = 1; kernel < 4; kernel ++)
    {
        passed_tests = 0;
        failed_tests = 0;
        //Loop for N
        for(int n = 1; n <= n_max; n++)
        {
            //Loop for W
            for(int w = 1; w <= w_max; w++)
            {
                //Check if N is a multiple of W
                if(n % w == 0)
                {
                    //Run test
                    if(run_test(n, w, kernel) == 0)
                    {
                        passed_tests++;
                    }
                    else
                    {
                        failed_tests++;
                    }
                }
            }
        }
        //display results for each kernel
        printf("##################KERNEL %d RESULTS###############\n", kernel);
        printf("PASSED TESTS: %d\n", passed_tests);
        printf("FAILED TESTS: %d\n", failed_tests);
        printf("###############################################\n");
        if(failed_tests == 0)
        {
            kernel_passed[kernel - 1] = 1;
        }
    }
    //display final results
    printf("##################FINAL RESULTS###############\n");
    for(int i = 0; i < 3; i++)
    {
        if(kernel_passed[i] == 1)
        {
            printf("KERNEL %d PASSED ALL TESTS\n", i + 1);
        }
        else
        {
            printf("KERNEL %d FAILED SOME TESTS\n", i + 1);
        }
    }
    printf("###############################################\n");
    
}

void exhaustive_bench()
{
    /*N={512,1024,2048,4096} y W={4,8,16,32}*/
    int N[4] = {512, 1024, 2048, 4096};
    int W[4] = {4, 8, 16, 32};
    int kernel[3] = {SIMPLY_MULTIPLY, COALESCED_MULTIPLY, SHARED_AB_MULTIPLY};

    //loop through all possible combinations (first kernel, then N, then W)
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 4; j++){
            for (int k = 0; k < 4; k++){
                run_test(N[j], W[k], kernel[i]);
            }
        }
    }
    
}

//main function
int main(int argc, char **argv){
    exhaustive_test();
    //exhaustive_bench();
}