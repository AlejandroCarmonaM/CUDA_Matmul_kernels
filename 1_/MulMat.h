#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h> // Añadir la biblioteca matemática para usar fabs
//Definir DEBUG para imprimir las matrices
//#define DEBUG

// mm_blo
//  A = Matriz A, B = Matriz B, C = Matriz C, matrix_size (ldm) = tamaño de la matriz,
// num_threads_t = número de hilos, num_filas_f = número de filas consecutivas
void mm_blo(float *A, float *B, float *C, int m, int k, int aux, int tam_blo_b, int num_hilos)
{
    int ldm = k;
    int fila, col, fila_bloque, col_bloque, l;
    float sum;

    int iam;
    omp_set_num_threads(num_hilos);
    //private -> cada hilo tiene su propia copia de la variable
    //collapse(2) -> colapsa los 2 primeros bucles anidados en un solo bucle
    //schedule(static, 1) -> divide el trabajo en bloques de tamaño 1
    #pragma omp parallel for private(fila, col, fila_bloque, col_bloque, l, sum) collapse(2) schedule(static, 1)
    for (fila = 0; fila < m; fila += tam_blo_b)
    {
        for (col = 0; col < aux; col += tam_blo_b)
        {
            #if defined (_OPENMP) 
            iam = omp_get_thread_num();
            #endif
            #ifdef DEBUG
            //printf("Soy el Hilo %d haciendo el BLOQUE (%d, %d)\n", iam, fila, col);
            #endif
            for (fila_bloque = fila; fila_bloque < fila + tam_blo_b; fila_bloque++)
            {
                for (col_bloque = col; col_bloque < col + tam_blo_b; col_bloque++)
                {
                    sum = 0.0;
                    for (l = 0; l < ldm; l++)
                    {
                        sum += A[fila_bloque * ldm + l] * B[l * aux + col_bloque];
                    }
                    C[fila_bloque * aux + col_bloque] = sum;
                }
            }
        }

    }

}