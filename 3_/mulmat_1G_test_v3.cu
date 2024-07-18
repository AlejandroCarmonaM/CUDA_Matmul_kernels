/*USAGE: compara_kernels –N=<dim_mat> -W=<dim_bloq> -K=<kernel>*/
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
    /*6. Call the kernel supertile*/
    //kernel_run_chrono(int N, int K, dim3 grid, dim3 block, float *A_d, float *B_d, float *C_d, int tile_dim)
    kernel_run_chrono(N, K, grid, block, A_d, B_d, C_d, tile_dim);

    /*7. Copy matrix C from the device to the host*/
    checkCudaErrors(cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    /*8. Check the result using a sequential algorithm*/
    int result = matrix_check_v3(A, B, C, M, N, K, W);

    if(result == 0)
    {
        printf("TEST PASSED: M=%d, N=%d, K=%d, W=%d\n", M, N, K, W);
    }
    else
    {
        printf("TEST FAILED: M=%d, N=%d, K=%d, W=%d\n", M, N, K, W);
        exit (EXIT_FAILURE);
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

    int n_max = 32;
    int m_max = 32;
    int k_max = 32;
    int w_max = 32;
    int kernel_passed = 0; //Array to store if a kernel passed all tests
    //int passed_tests_kernel[3] = {0, 0, 0}; //Array to store the number of passed tests for each kernel

    //We will test all possible (allowed by the restrictions) combinations of M, N, K and W
    int passed_tests = 0;
    int failed_tests = 0;
    for (int m = 1; m <= m_max; m++){
        for (int n = 1; n <= n_max; n++){
            for (int k = 1; k <= k_max; k++){
                for (int w = 1; w <= w_max; w++){
                    if (m % w == 0 && n % w == 0 && k % w == 0){
                        if (run_test(m, n, k, w) == 0){
                            passed_tests++;
                        }
                        else{
                            exit (EXIT_FAILURE);
                            failed_tests++;
                        }
                    }
                }
            }
        }
    }

    //display results for each kernel
    printf("##################KERNEL RESULTS###############\n");
    printf("PASSED TESTS: %d\n", passed_tests);
    printf("FAILED TESTS: %d\n", failed_tests);
    printf("###############################################\n");
    if(failed_tests == 0)
    {
        kernel_passed = 1;
        //display if kernel passed all tests
        printf("KERNEL PASSED ALL TESTS: %d\n", kernel_passed);
    }
    else
    {
        kernel_passed = 0;
        //display if kernel failed any tests
        printf("KERNEL FAILED TESTS: %d\n", kernel_passed);
    }
    
}


//main function
int main(int argc, char **argv){
    exhaustive_test();
}