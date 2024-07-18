#include "MulMat.h"
//include cublas
#include <cublas_v2.h>
#define NUM_HILOS 8
//Print matrix function
void print_matrix(float *matrix, int m, int n)
{
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            printf("%f ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

void initialize_matrices(int M, int K, int N, float** A, float** B, float** C) {
    /*1. Allocate memory for matrices A, B and C*/
    *A = (float *) malloc(M * K * sizeof(float));
    *B = (float *) malloc(K * N * sizeof(float));
    *C = (float *) malloc(M * N * sizeof(float));

    /*2. Initialize matrices A and B with random values*/
    for (int i = 0; i < M * K; i++){
        (*A)[i] = (float) rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++){
        (*B)[i] = (float) rand() / RAND_MAX;
    }
}

void allocate_device_matrices(int M, int K, int N, float** A_d, float** B_d, float** C_d) {
    //printf("M: %d\n", M);
    /* Allocate memory for matrices A, B and C in the device */
    checkCudaErrors(cudaMalloc((void **) A_d, M * K * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) B_d, K * N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) C_d, M * N * sizeof(float)));
}

//Sequential matrix multiplication
void seq_mult(float *A, float *B, float *C_check, int m, int n, int k)
{
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            C_check[i * n + j] = 0.0f;
            for (int l = 0; l < k; l++){
                C_check[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

int matrix_check(float *A, float *B, float *C, int N, int W, int kernel)
{
    float *C_check = (float *) malloc(N * N * sizeof(float));
    //seq_mult(A, B, C_check, N, N, W);
    //a,b,c,m,k,n
    mm_blo(A, B, C_check, N, W, N, W, NUM_HILOS);
    for (int i = 0; i < N * N; i++){
        if (abs(C[i] - C_check[i]) > 1e-3){
            //Display configuration (N, W, K)
            printf("ERROR IN: N: %d, W: %d, K: %d\n", N, W, kernel);
            return 1;
        }
    }
    //clean up
    free(C_check);
    return 0;
}

int matrix_check_v2(float *A, float *B, float *C, int N, int W)
{
    float *C_check = (float *) malloc(N * N * sizeof(float));
    //seq_mult(A, B, C_check, N, N, N);
    mm_blo(A, B, C_check, N, N, N, W, NUM_HILOS);
    for (int i = 0; i < N * N; i++){
        if (abs(C[i] - C_check[i]) > 1e-3){
            //Display configuration (N, W, K)
            printf("ERROR IN: N: %d, W: %d\n", N, W);
            return 1;
        }
    }

    //free newly allocated memory
    free(C_check);
    return 0;
}

int matrix_check_v3(float *A, float *B, float *C, int M, int N, int K, int W)
{
    /*8. Check the result using a sequential algorithm*/
    float *C_check = (float *) malloc(M * N * sizeof(float));
    //seq_mult(A, B, C_check, M, N, K);
    mm_blo(A, B, C_check, M, K, N, W, NUM_HILOS);
    for (int i = 0; i < M * N; i++){
        if (abs(C[i] - C_check[i]) > 1e-3){
            //Display error for configuration (M, N, K, W)
            printf("Error in configuration (M = %d, N = %d, K = %d, W = %d)\n", M, N, K, W);
            //Display matrices A, B and C
            printf("Matrix A\n");
            print_matrix(A, M, K);
            printf("Matrix B\n");
            print_matrix(B, K, N);
            printf("Matrix C\n");
            print_matrix(C, M, N);
            printf("Matrix C_check\n");
            print_matrix(C_check, M, N);
            return 1;
        }
    }

    //free newly allocated memory
    free(C_check);
    return 0;
}

int matrix_check_v4(float *A, float *B, float *C, int M, int N, int K, int W, int F)
{
    /*8. Check the result using a sequential algorithm*/
    float *C_check = (float *) malloc(M * N * sizeof(float));
    //seq_mult(A, B, C_check, M, N, K);
    mm_blo(A, B, C_check, M, K, N, W, NUM_HILOS);
    for (int i = 0; i < M * N; i++){
        if (abs(C[i] - C_check[i]) > 1e-3){
            //Display error for configuration (M, N, K, W)
            printf("Error in configuration (M = %d, N = %d, K = %d, W = %d, F=%d)\n", M, N, K, W, F);
            //display difference
            printf("Difference: %f\n", abs(C[i] - C_check[i]));
            //print elements
            printf("C[%d]: %f\n", i, C[i]);
            printf("C_check[%d]: %f\n", i, C_check[i]);
            //Display matrices A, B and C
            /*printf("Matrix A\n");
            print_matrix(A, M, K);
            printf("Matrix B\n");
            print_matrix(B, K, N);
            printf("Matrix C\n");
            print_matrix(C, M, N);
            printf("Matrix C_check\n");
            print_matrix(C_check, M, N);*/
            return 1;
        }
    }

    //free newly allocated memory
    free(C_check);
    return 0;
}



//checkCudaErrors(cudaMemcpy(A_d, A, N * W * sizeof(float), cudaMemcpyHostToDevice));
int matrix_device_copy(float *A, float *B, float *A_d, float *B_d, int N, int W)
{
    //Copy A, B and C to device
    checkCudaErrors(cudaMemcpy(A_d, A, N * W * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B_d, B, W * N * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

int matrix_device_copy_mnk(float *A, float *B, float *A_d, float *B_d, int M, int N, int K)
{
    //Copy A, B and C to device
    checkCudaErrors(cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

int cublas_matrix_check(float* A, float* B, float* C, int M, int N, int K, int W, int F) {
    //Copy from the CPU to the GPU
    float *A_d, *B_d, *C_d;
    checkCudaErrors(cudaMalloc((void **) &A_d, M * K * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &B_d, K * N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &C_d, M * N * sizeof(float)));

    checkCudaErrors(cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B_d, N, A_d, K, &beta, C_d, N);
    cublasDestroy(handle);

    //Copy from the GPU to the CPU
    float* C_check = (float*)malloc(M * N * sizeof(float));
    checkCudaErrors(cudaMemcpy(C_check, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < M * N; i++){
        //for this comparison, we use a tolerance of 1e-2
        if (abs(C[i] - C_check[i]) > 1e-2){
            //Display error for configuration (M, N, K, W)
            printf("Test failed for M = %d, N = %d, K = %d, W = %d, F = %d\n", M, N, K, W, F);
            //Display difference
            //printf("Difference: %f\n", abs(C[i] - C_check[i]));
            //Display matrices A, B and C
            /*printf("Matrix A\n");
            print_matrix(A, M, K);
            printf("Matrix B\n");
            print_matrix(B, K, N);
            printf("Matrix C\n");
            print_matrix(C, M, N);
            printf("Matrix C_check\n");
            print_matrix(C_check, M, N);*/
            return 1;
        }
    }
    free(C_check);
    //Free GPU memory
    checkCudaErrors(cudaFree(A_d));
    checkCudaErrors(cudaFree(B_d));
    checkCudaErrors(cudaFree(C_d));
    return 0;
}