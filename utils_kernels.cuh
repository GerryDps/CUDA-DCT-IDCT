#ifndef PROJECT_CUDA_UTILS_KERNELS_CUH
#define PROJECT_CUDA_UTILS_KERNELS_CUH

// Kernel CUDA per la sottrazione element-wise matrice - scalare
__global__ void sub_matrix_scalar(const float* A, const float scalar, float* C, const int size);

// Kernel CUDA per l'addizione element-wise matrice - scalare
__global__ void add_matrix_scalar(const float* A, const float scalar, float* C, const int size);

// Kernel CUDA per la divisione elemento per elemento
__global__ void divide_matrices(const float* A, const float* B, float* C, const int size);

// Kernel CUDA per la moltiplicazione elemento per elemento
__global__ void multiply_matrices(const float* A, const float* B, float* C, const int size);

// Kernel CUDA per la creazione della matrice di trasformazione multipli di 8
__global__ void upgrade_T_matrix(const float* T, float* C, const size_t size);

#endif //PROJECT_CUDA_UTILS_KERNELS_CUH
