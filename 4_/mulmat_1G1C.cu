/*USAGE: compara_kernels -N=<dim_mat> -W=<dim_bloq> -K=<kernel>*/
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
#define DEBUG

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

    //8.1 print Header
    printf("M,N,K,W,F,GPU_KERNEL_TIME,GPU_TRANSFER_TIME,GPU_TIME,CPU_TIME,TOTAL_TIME\n");
    float gpu_time = times[GPU_KERNEL_TIME] + times[GPU_TRANSFER_TIME];
    float total_time;
    if(gpu_time > times[CPU_TIME])
    {
        total_time = gpu_time;
    }
    else
    {
        total_time = times[CPU_TIME];
    }
    printf("%d,%d,%d,%d,%d,%f,%f,%f,%f,%f\n", M, N, K, W, F, times[GPU_KERNEL_TIME], times[GPU_TRANSFER_TIME], gpu_time, times[CPU_TIME], total_time);
   

    //print matrices A, B and C
    printf("Matrix A\n");
    print_matrix(A, M, K);
    printf("Matrix B\n");
    print_matrix(B, K, N);
    printf("Matrix C\n");
    print_matrix(C, M, N);

    // Check the result
    #ifdef DEBUG
        int result = matrix_check_v4(A, B, C, M, N, K, W, F);
        if(result == 0)
        {
            printf("Test passed for M = %d, N = %d, K = %d, W = %d, F = %d\n", M, N, K, W, F);
        }
        else
        {
            printf("TEST FAILED: M=%d, N=%d, K=%d, W=%d\n", M, N, K, W);
        }
    #endif
    // Free memory
    free(A);
    free(B);
    free(C);
    return result;
}

//main function
int main(int argc, char **argv){
     /*1. parse arguments: mulmat_1G -M=<dim_mat> -N=<dim_mat> -K=<dim_mat> -W=<dim_bloq> -F=<filas_cpu>*/
    int M, N, K, W, F;
    if (argc != 6){
        printf("Usage: mulmat_1G1C -M=<dim_mat> -N=<dim_mat> -K=<dim_mat> -W=<dim_bloq> -F=<filas_cpu>\n");
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
        if (sscanf(argv[i], "-F=%d", &F) == 1){
            continue;
        }
    }
    //Check if M, N and K are multiples of W
    if (M % W != 0 || N % W != 0 || K % W != 0 || F % W != 0){
        printf("M, N, K and F must be multiples of W\n");
        return 0;
    }
    if (W > 32){
        printf("W must be less than 32\n");
        return 1;
    }
    

    //run the test
    run_test(M, N, K, W, F);
    return 0;

    
}