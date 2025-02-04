# CUDA_Matmul_kernels

## Description

This project implements matrix multiplication (matmul) using CUDA, demonstrating the performance available when leveraging GPU computation and Shared Memory usage. The primary focus is on optimizing kernel execution time.

It also includes OpenMP use to leverage CPU too if necessary.

Full documentation with performance evaluation only available in Spanish.

## Features

- **Hybrid Kernel Execution**: Combines CPU (OpenMP) and GPU (CUDA) computation to maximize efficiency.
- **Measured Data Transfer**: Records data transfer times between host and device.
- **Multiple Kernel Implementations**: Includes various kernels for rectangular, squared and general matrices.
- **Shared Memory Usage**: Kernels with Shared Memory support delegate their threadblocks to load their respective tiles to Shared Memory to increase performance.
- **Performance Comparison**: Compares execution times across different kernel configurations and matrix sizes.
- **Validation Tests**: Ensures correctness of results with test programs.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/AlejandroCarmonaM/CUDA_Matmul_kernels.git
    cd CUDA_Matmul_kernel
    ```

2. **Set Up CUDA**:

    Ensure that you have the CUDA toolkit installed. You can download it from the NVIDIA website.

## Usage

Running the matrix multiplication:
- Navigate to the chosen implementation.
- Modify the `Makefile` argument called `PROG` and Run the make command to build whichever version of a given implementation (regular, benchmark, test on CPU, test on GPU (cublas)).
- Run the executable.

## Customizing Parameters

You can modify the matrix dimensions and block configurations when running the program via CLI.

## Validating Results

If the implementation is modified, use the provided test programs to validate the correctness of your kernel implementations:

- 1. Modify the `Makefile` argument called `PROG` and write the selected test
```bash
make
./selected_test
```

## Kernel Comparison Program

A program called `compara_kernels` allows comparison of the three already existent kernel versions of matrix multiplication described in class: `simpleMultiply`, `coalescedMultiply`, and `sharedABMultiply`. This program is designed to multiply rectangular matrices of real numbers in the form CNxN = ANxWBWxN, using the kernel chosen by the user. The configuration parameters are set as follows:

- Block of threads: WxW.
- Block (tile) of data: WxW.
- Grid: TxT blocks of threads.
- Each thread computes an element of C from a row of A and a column of B.
- Each block of threads computes the elements of a tile of C from multiplying a tile of A and a tile of B.

Example syntax:

```bash
compara_kernels –N=<matrix_dimensions> -W=<block_dimensions> -K=<kernel>
```

## Single GPU Matrix Multiplication Program

A program called `mulmatcua_1G` calculates the product of two square matrices of real numbers in the form CNxN=ANxNBNxN, using the `sharedMultiply` kernel on a GPU.

Example syntax:

```bash
mulmatcua_1G –N=<matrix_dimensions> -W=<block_dimensions>
```
For simplicity, N must be a multiple of W.

## General GPU Matrix Multiplication Program

A program called `mulmat_1G` calculates the product of two matrices of real numbers in the form CMxN=AMxKBKxN, using the `sharedMultiply` kernel on a GPU.

Example syntax:

```bash
mulmat_1G –M=<matrix_dimensions> –N=<matrix_dimensions> –K=<matrix_dimensions> -W=<block_dimensions>
```

For simplicity, M, N and K must be multiples of W.

## Hybrid CPU-GPU Matrix Multiplication Program

A program called `mulmat_1C1G` uses both OpenMP and CUDA to calculate the product of two matrices of real numbers in the form CMxN=AMxKBKxN, using the `sharedMultiply` kernel. The CPU handles the computation of the last F rows of matrix C, while the GPU handles the first M-F rows of C.

Example syntax:

```bash
mulmat_1C1G –M=<matrix_dimensions> –N=<matrix_dimensions> –K=<matrix_dimensions> -W=<block_dimensions> -F=<cpu_rows>
```

Same restrictions apply to the GPU part with M, N, K and F as multiples to W.
