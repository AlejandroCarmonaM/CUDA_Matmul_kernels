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
#define WARMUP_ITERS 10
#define BENCH_ITERS 10


/**************REQUIREMENTS**********************/
/*C_MxN=A_MxK B_KxN*/
/*M, N, K multiple of W */
/*Grid size: SxT (S=M/W, T=N/W)*/
/*Threads per Block: WxW (the number of threads per block will be the leading dimension of the tile)*/
/*Tile size: WxW*/
/*R=K/W*/

//GLOBAL VARIABLES
double gpu_kernel_time = 0.0;
double gpu_transfer_time = 0.0;


////////////////////////////////////PROGRAM FUNCTIONS////////////////////////////////////

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
            if(F > 0){
                start = omp_get_wtime();
                mm_blo(A_cpu_ptr, B, C_cpu_ptr, F, K, N, W, NUM_THREADS);
                stop = omp_get_wtime();
                cpu_time = stop - start;
            }
        }
    }
    
    times[GPU_KERNEL_TIME] = gpu_kernel_time;
    times[GPU_TRANSFER_TIME] = gpu_transfer_time;
    times[CPU_TIME] = cpu_time;
}



void run_test(int M, int N, int K, int W, int F)
{
    // Create matrices A, B and C
    float *A, *B, *C;
    initialize_matrices(M, K, N, &A, &B, &C);
    //array to store times of a certain iteration
    double *times = (double *)malloc(3 * sizeof(double));
    //array to store total times of all iterations
    double *total_times = (double *)malloc(3 * sizeof(double));
    
    //Warmup
    for(int i = 0; i < WARMUP_ITERS; i++)
    {
        hybrid_mm(A, B, C, M, N, K, W, F, times);
    }
    //Benchmark
    for(int i = 0; i < BENCH_ITERS; i++)
    {
        hybrid_mm(A, B, C, M, N, K, W, F, times);
        total_times[GPU_KERNEL_TIME] += times[GPU_KERNEL_TIME];
        total_times[GPU_TRANSFER_TIME] += times[GPU_TRANSFER_TIME];
        total_times[CPU_TIME] += times[CPU_TIME];
    }
    //Average times
    total_times[GPU_TRANSFER_TIME] /= BENCH_ITERS;
    total_times[GPU_KERNEL_TIME] /= BENCH_ITERS;
    total_times[CPU_TIME] /= BENCH_ITERS;

    // print values for header: printf("M,N,K,W,F,GPU_KERNEL_TIME,GPU_TRANSFER_TIME,GPU_TIME,CPU_TIME,TOTAL_TIME\n");
    float gpu_time = total_times[GPU_KERNEL_TIME] + total_times[GPU_TRANSFER_TIME];
    float total_time;
    if(gpu_time > total_times[CPU_TIME])
    {
        total_time = gpu_time;
    }
    else
    {
        total_time = total_times[CPU_TIME];
    }
    printf("%d,%d,%d,%d,%d,%f,%f,%f,%f,%f\n", M, N, K, W, F, total_times[GPU_KERNEL_TIME], total_times[GPU_TRANSFER_TIME], gpu_time, total_times[CPU_TIME], total_time);
   
    // Free memory
    free(A);
    free(B);
    free(C);
}
/*Compara razonadamente los tiempos de ejecución de la multiplicación completa, para
F={0,4,8,12,16,20,24,28,32}, con N=M=K=2048 y W=4.*/
void exhaustive_bench()
{
    int M = 2048;
    int N = 2048;
    int K = 2048;
    int W = 4;
    int F;
    printf("M,N,K,W,F,GPU_KERNEL_TIME,GPU_TRANSFER_TIME,GPU_TIME,CPU_TIME,TOTAL_TIME\n");
    for(F = 0; F <= 32; F += 4)
    {
        run_test(M, N, K, W, F);
    }
    
}

//main function
int main(int argc, char **argv){
    exhaustive_bench();
}