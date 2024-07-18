#include "MulMat.h"
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
    mm_blo(A, B, C_check, N, W, N, W, N);
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

//checkCudaErrors(cudaMemcpy(A_d, A, N * W * sizeof(float), cudaMemcpyHostToDevice));
int matrix_device_copy(float *A, float *B, float *A_d, float *B_d, int N, int W)
{
    //Copy A, B and C to device
    checkCudaErrors(cudaMemcpy(A_d, A, N * W * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B_d, B, W * N * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}
