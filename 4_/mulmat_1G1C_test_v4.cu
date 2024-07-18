/*USAGE: compara_kernels –N=<dim_mat> -W=<dim_bloq> -K=<kernel>*/
// includes, system
#include <omp.h>
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

////////////////////////////////////////////////////////////////////////////////
// kernel includes
#include "sharedABMultiply_kernel_hyper_tile.cu"

#include "utils.h"

////////////////////////////////////////////////////////////////////////////////

#define NUM_THREADS 8
#define GPU_KERNEL_TIME 0
#define GPU_TRANSFER_TIME 1
#define CPU_TIME 2

//GLOBAL VARIABLES
double gpu_kernel_time = 0.0;
double gpu_transfer_time = 0.0;


/**************REQUIREMENTS**********************/
/*C_MxN=A_MxK B_KxN*/
/*M, N, K multiple of W */
/*Grid size: SxT (S=M/W, T=N/W)*/
/*Threads per Block: WxW (the number of threads per block will be the leading dimension of the tile)*/
/*Tile size: WxW*/
/*R=K/W*/

void hybrid_kernel_run_chrono(int M_gpu, int K, int N, float *A, float *B, float *C, dim3 grid, dim3 block, int tile_dim) {
    // Allocate memory for matrices A_d, B_d and C_d in the device
    float *A_d, *B_d, *C_d;
    allocate_device_matrices(M_gpu, K, N, &A_d, &B_d, &C_d);
    
    // Recording events GPU
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record event: matrix A (partially) and B copy to the device
    cudaEventRecord(start, 0);
    checkCudaErrors(cudaMemcpy(A_d, A, M_gpu * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    gpu_transfer_time = milliseconds / 1000.0;
    
    cudaEventRecord(start, 0);
    sharedABMultiply<<<grid, block, 2 * tile_dim * tile_dim * sizeof(float)>>>(A_d, B_d, C_d, N, K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    gpu_kernel_time = milliseconds / 1000.0;

    // Copy the partial result from the device to the host
    cudaEventRecord(start, 0);
    checkCudaErrors(cudaMemcpy(C, C_d, M_gpu * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    gpu_transfer_time += milliseconds / 1000.0;

    //GPU Cleanup
    checkCudaErrors(cudaFree(A_d));
    checkCudaErrors(cudaFree(B_d));
    checkCudaErrors(cudaFree(C_d));
}

void hybrid_mm(float *A, float *B, float *C, int M, int N, int K, int W, int F, double *times)
{
    // GPU Parameter setup
    int T = N/W;
    int S = M/W;
    dim3 grid(T, S); //Grid size: SxT
    dim3 block(W, W); //Threads per Block: WxW
    int tile_dim = W; //Tile size: WxW
    // M_gpu = M-F; M_cpu = F
    int M_gpu = M - F;

    
    //CPU Parameter setup
    //We must advance the pointer of A and C to the F-th row
    float *A_cpu_ptr = A + (M_gpu) * K;
    float *C_cpu_ptr = C + (M_gpu) * N;

    double cpu_time = 0.0;
    double start, stop;

    //Allow nested parallelism so that the main thread can run the GPU kernel
    //and one thread can then create more threads to run the CPU kernel
    omp_set_nested(1);
    omp_set_num_threads(2);

    // Copy the matrices A (partially) and B to the device
    #pragma omp parallel
    {
        int iam = omp_get_thread_num();
        if (iam == 0) {
            hybrid_kernel_run_chrono(M_gpu, K, N, A, B, C, grid, block, tile_dim);

        } else {
            // The rest of the threads execute the rest of the matrix
            start = omp_get_wtime();
            mm_blo(A_cpu_ptr, B, C_cpu_ptr, F, K, N, W, NUM_THREADS-1);
            stop = omp_get_wtime();
            cpu_time = stop - start;
        }
    }
    
    times[GPU_KERNEL_TIME] = gpu_kernel_time;
    times[GPU_TRANSFER_TIME] = gpu_transfer_time;
    times[CPU_TIME] = cpu_time;
}



int run_test(int M, int N, int K, int W, int F)
{
    // Create matrices A, B and C
    float *A, *B, *C;
    initialize_matrices(M, K, N, &A, &B, &C);
    // M_gpu = M-F; M_cpu = F
    int M_gpu = M - F;
    //times array of doubles to store the times of the GPU kernel, GPU transfer and CPU kernel
    double times[3] = {0.0, 0.0, 0.0};

    if(M_gpu == 0)
    {
        //mm_blo(float *A, float *B, float *C, int m, int k, int n, int tam_blo_b, int num_hilos)
        mm_blo(A, B, C, M, K, N, W, F);

    }
    else{
        //call hybrid_mm
        hybrid_mm(A, B, C, M, N, K, W, F, times);
    }
    
    // Check the result
    //int matrix_check_v3(float *A, float *B, float *C, int M, int N, int K, int W)
    int result = matrix_check_v4(A, B, C, M, N, K, W, F);
    if(result == 0)
    {
        printf("Test passed for M = %d, N = %d, K = %d, W = %d, F = %d\n", M, N, K, W, F);
    }
    else
    {
        exit(EXIT_FAILURE);
    }
    // Free memory
    free(A);
    free(B);
    free(C);
    return result;
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
    int f_max = 5;
    int kernel_passed = 0; //Array to store if a kernel passed all tests
    //int passed_tests_kernel[3] = {0, 0, 0}; //Array to store the number of passed tests for each kernel

    //We will test all possible (allowed by the restrictions) combinations of M, N, K and W
    int passed_tests = 0;
    int failed_tests = 0;
    for (int m = 1; m <= m_max; m++){
        for (int n = 1; n <= n_max; n++){
            for (int k = 1; k <= k_max; k++){
                for (int w = 1; w <= w_max; w++){
                    for (int f = 1; f <= f_max; f++){
                        //Check if M, N, K and F are multiples of W
                        if (m % w == 0 && n % w == 0 && k % w == 0 && f % w == 0 && m >= f){
                            int result = run_test(m, n, k, w, f);
                            if (result == 0){
                                passed_tests++;
                            }
                            else{
                                failed_tests++;
                            }
                        }
                    }
                }
            }
        }
    }
    /*int result = run_test(8, 2, 4, 2, 2);
    if (result == 0){
        passed_tests++;
    }
    else{
        failed_tests++;
    }*/

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