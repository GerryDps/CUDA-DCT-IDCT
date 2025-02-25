#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "utils_kernels.cuh"

// Kernel CUDA per la sottrazione element-wise matrice - scalare
__global__ void sub_matrix_scalar(const float* A, const float scalar, float* C, const int size) {
    // Calcola l'indice globale del thread
    const int Id_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int Id_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int global = Id_y * gridDim.x * BLOCK_SIZE + Id_x;

    // Controlla che l'indice sia all'interno dei limiti
    if (global < size) {
        C[global] =  A[global] - scalar;
    }
}

// Kernel CUDA per l'addizione element-wise matrice - scalare
__global__ void add_matrix_scalar(const float* A, const float scalar, float* C, const int size) {
    // Calcola l'indice globale del thread
    const int Id_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int Id_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int global = Id_y * gridDim.x * BLOCK_SIZE + Id_x;

    // Controlla che l'indice sia all'interno dei limiti
    if (global < size) {
        C[global] =  A[global] + scalar;
    }
}

// Kernel CUDA per la divisione elemento per elemento
__global__ void divide_matrices(const float* A, const float* B, float* C, const int size) {
    // Calcola l'indice globale del thread
    const int Id_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int Id_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int global = Id_y * gridDim.x * BLOCK_SIZE + Id_x;

    // Controlla che l'indice sia all'interno dei limiti
    if (global < size) {
        C[global] =  round(A[global] / B[threadIdx.y * BLOCK_SIZE + threadIdx.x]);
    }
}

// Kernel CUDA per la moltiplicazione elemento per elemento
__global__ void multiply_matrices(const float* A, const float* B, float* C, const int size) {
    // Calcola l'indice globale del thread
    const int Id_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int Id_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int global = Id_y * gridDim.x * BLOCK_SIZE + Id_x;

    // Controlla che l'indice sia all'interno dei limiti
    if (global < size) {
        C[global] =  A[global] * B[threadIdx.y * BLOCK_SIZE + threadIdx.x];
    }
}

// Kernel CUDA per la creazione della matrice di trasformazione multipli di 8
__global__ void upgrade_T_matrix(const float* T, float* C, const size_t size) {
    // Calcola l'indice globale del thread
    const int Id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int Id_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int global = Id_y * gridDim.x * blockDim.x + Id_x;
    if(global>=size*size)return;
    const int se = (blockIdx.x==blockIdx.y);
    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    C[global] = se * T[index];
}