//%%writefile benchmark_newAppr.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Check CUDA error
#define CHECK_CUDA(call)                                          \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            printf("%s : %d", cudaGetErrorString(err), __LINE__); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    }

#define BLOCK_SIZE 8

__constant__ float const_quant_matrix[BLOCK_SIZE*BLOCK_SIZE];

// Kernels CUDA per le operazioni aritmetiche element-wise
__global__ void sub_matrix_scalar(const float* A, const float scalar, float* C, const int size);
__global__ void add_matrix_scalar(const float* A, const float scalar, float* C, const int size);

__global__ void divide_matrices(const float* A, const float* B, float* C, const int size);
__global__ void multiply_matrices(const float* A, const float* B, float* C, const int size);

// Using cuda kernels to compute the DCT and the IDCT
// Those FUNC use intenal dct/idct kernel function, differs according to implementation used.
void dct_all_blocks_cuda(float* image_matrix, const int img_height, const int img_width, const float* transform_matrix, float* result);
void idct_all_blocks_cuda(const float* image_matrix, const int img_height, const int img_width, const float* transform_matrix, float* result);

int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("Use: %s <width/height>\n", argv[0]);
        return 1;
    }
    size_t input = strtoul(argv[1], NULL, 10);
    size_t width, height;
    width = input;
    height = input;

    float* image_matrix_float;
    image_matrix_float = (float*)malloc(width * height * sizeof(float));
    srand(42);
    for (int i = 0; i < height;i++) {
        for (int j = 0; j < width; j++) {
            image_matrix_float[i * width + j] = rand() % 256;;
        }
    }

    // Quantization matrix (su constant)
    float quant_matrix[BLOCK_SIZE * BLOCK_SIZE] = {
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99};
    // memoria costante
    CHECK_CUDA(cudaMemcpyToSymbol(const_quant_matrix, quant_matrix, sizeof(quant_matrix)));

    // Transform matrix (hardcoded for simplicity)
    float transform_matrix[BLOCK_SIZE * BLOCK_SIZE] = {
            0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339,
            0.5, 0.5, 0, 0, 0, 0, -0.5, -0.5,
            0.4472136, 0.2236068, -0.2236068, -0.4472136, -0.4472136, -0.2236068, 0.2236068, 0.4472136,
            0, 0, -0.70710678, 0, 0, 0.70710678, 0, 0,
            0.35355339, -0.35355339, -0.35355339, 0.35355339, 0.35355339, -0.35355339, -0.35355339, 0.35355339,
            0.5, -0.5, 0, 0, 0, 0, 0.5, -0.5,
            0.2236068, -0.4472136, 0.4472136, -0.2236068, -0.2236068, 0.4472136, -0.4472136, 0.2236068,
            0, 0, 0, -0.70710678, 0.70710678, 0, 0, 0};

    // allocate host memory for the result image
    float *result;
    result = (float *)malloc(width * height * sizeof(float));

    // allocate device memory for: image_block (d_A), trasform_matrix (d_B), and result (d_C)
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, width * height * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, width * height * sizeof(float)));

    // copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_A, image_matrix_float, width * height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, transform_matrix, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Compute DCT
    // d_A = image_block ; d_B = transform_matrix ; d_C = result
    dct_all_blocks_cuda(d_A, height,width,d_B, d_C);

    // copy device memory to host
    // result = d_C
    CHECK_CUDA(cudaMemcpy(result, d_C, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    // d_E = result of the idct applied on the block_image compressed
    float *d_E;
    CHECK_CUDA(cudaMalloc(&d_E, width * height * sizeof(float)));

    // Compute the idct
    // d_C = result ; d_B = transform_matrix ; d_E = result
    idct_all_blocks_cuda(d_C, height, width, d_B,d_E);

    // copy device memory to host
    // result = d_E
    CHECK_CUDA(cudaMemcpy(result, d_E, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    //CLeanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_E));
    free(result);
    free(image_matrix_float);
    return 0;
}

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

/* *
 * Effettua la DCT utilizzando la matrice di trasformazione
 * (TRASFORM_MATRIX @ IMAGE_MATRIX) @ TRANSFORM_MATRIX.T
 * La matrice di trasformazione è 8x8
 * result = TRASFORM_MATRIX @ IMAGE_MATRIX
 * result = result @ TRANSFORM_MATRIX.T
 * */
__global__ void cuda_matrix_dct(const float* image_matrix, const float* transform_matrix, float* result) {
    __shared__ float shared_matrix[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float shared_transform[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float shared_image[BLOCK_SIZE*BLOCK_SIZE];

    // Calcola l'indice globale del thread
    const int Id_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int Id_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int global = Id_y * gridDim.x * BLOCK_SIZE + Id_x

    float sums = 0;
    shared_transform[threadIdx.y * BLOCK_SIZE + threadIdx.x] = transform_matrix[threadIdx.y * BLOCK_SIZE + threadIdx.x];
    shared_image[threadIdx.y * BLOCK_SIZE + threadIdx.x] = image_matrix[global];
    __syncthreads();

    // result = transform_matrix @ image_matrix
    // result = transform_matrix[righe] @ image_matrix[colonne]
    sums += shared_transform[threadIdx.y * BLOCK_SIZE + 0] * shared_image[threadIdx.x + 0 * BLOCK_SIZE];
    sums += shared_transform[threadIdx.y * BLOCK_SIZE + 1] * shared_image[threadIdx.x + 1 * BLOCK_SIZE];
    sums += shared_transform[threadIdx.y * BLOCK_SIZE + 2] * shared_image[threadIdx.x + 2 * BLOCK_SIZE];
    sums += shared_transform[threadIdx.y * BLOCK_SIZE + 3] * shared_image[threadIdx.x + 3 * BLOCK_SIZE];
    sums += shared_transform[threadIdx.y * BLOCK_SIZE + 4] * shared_image[threadIdx.x + 4 * BLOCK_SIZE];
    sums += shared_transform[threadIdx.y * BLOCK_SIZE + 5] * shared_image[threadIdx.x + 5 * BLOCK_SIZE];
    sums += shared_transform[threadIdx.y * BLOCK_SIZE + 6] * shared_image[threadIdx.x + 6 * BLOCK_SIZE];
    sums += shared_transform[threadIdx.y * BLOCK_SIZE + 7] * shared_image[threadIdx.x + 7 * BLOCK_SIZE];
    shared_matrix[threadIdx.y * BLOCK_SIZE + threadIdx.x] = sums;
    sums = 0;

    __syncthreads();

    // result = result(precedente) @ transform_matrix.T (trasposta)
    // result = result(precedente)[righe] @ transform_matrix[righe] (perchè la trasposta)
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 0] * shared_transform[threadIdx.x * BLOCK_SIZE + 0];
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 1] * shared_transform[threadIdx.x * BLOCK_SIZE + 1];
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 2] * shared_transform[threadIdx.x * BLOCK_SIZE + 2];
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 3] * shared_transform[threadIdx.x * BLOCK_SIZE + 3];
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 4] * shared_transform[threadIdx.x * BLOCK_SIZE + 4];
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 5] * shared_transform[threadIdx.x * BLOCK_SIZE + 5];
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 6] * shared_transform[threadIdx.x * BLOCK_SIZE + 6];
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 7] * shared_transform[threadIdx.x * BLOCK_SIZE + 7];
    result[Id_y * gridDim.x * BLOCK_SIZE + Id_x] = sums;
}

/* *
 * Effettua la IDCT utilizzando la matrice di trasformazione
 * (TRANSFORM_MATRIX.T @ DCT_MATRIX) @ TRANSFORM_MATRIX
 * La matrice di trasformazione è 8x8
 * result = TRANSFORM_MATRIX.T @ DCT_MATRIX
 * result = result @ TRANSFORM_MATRIX
 * */
__global__ void cuda_matrix_idct(const float* image_matrix, const float* transform_matrix, float* result) {
    __shared__ float shared_matrix[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float shared_transform[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float shared_image[BLOCK_SIZE*BLOCK_SIZE];
    // Calcola l'indice globale del thread
    const int Id_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int Id_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int global = Id_y * gridDim.x * BLOCK_SIZE + Id_x;

    float sums = 0;
    shared_transform[threadIdx.y * BLOCK_SIZE + threadIdx.x] = transform_matrix[threadIdx.y * BLOCK_SIZE + threadIdx.x];
    shared_image[threadIdx.y * BLOCK_SIZE + threadIdx.x] = image_matrix[global];
    __syncthreads();

    // result = transform_matrix.T @ dct_matrix
    // result = transform_matrix[colonne](x trasposta) @ image_matrix[colonne]
    sums += shared_transform[0 * BLOCK_SIZE + threadIdx.y] * shared_image[threadIdx.x + 0 * BLOCK_SIZE];
    sums += shared_transform[1 * BLOCK_SIZE + threadIdx.y] * shared_image[threadIdx.x + 1 * BLOCK_SIZE];
    sums += shared_transform[2 * BLOCK_SIZE + threadIdx.y] * shared_image[threadIdx.x + 2 * BLOCK_SIZE];
    sums += shared_transform[3 * BLOCK_SIZE + threadIdx.y] * shared_image[threadIdx.x + 3 * BLOCK_SIZE];
    sums += shared_transform[4 * BLOCK_SIZE + threadIdx.y] * shared_image[threadIdx.x + 4 * BLOCK_SIZE];
    sums += shared_transform[5 * BLOCK_SIZE + threadIdx.y] * shared_image[threadIdx.x + 5 * BLOCK_SIZE];
    sums += shared_transform[6 * BLOCK_SIZE + threadIdx.y] * shared_image[threadIdx.x + 6 * BLOCK_SIZE];
    sums += shared_transform[7 * BLOCK_SIZE + threadIdx.y] * shared_image[threadIdx.x + 7 * BLOCK_SIZE];
    shared_matrix[threadIdx.y * BLOCK_SIZE + threadIdx.x] = sums;
    sums = 0;
    __syncthreads();

    // result = result(precedente) @ transform_matrix
    // result = result[righe] @ transform_matrix[colonne]
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 0] * shared_transform[0 * BLOCK_SIZE + threadIdx.x];
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 1] * shared_transform[1 * BLOCK_SIZE + threadIdx.x];
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 2] * shared_transform[2 * BLOCK_SIZE + threadIdx.x];
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 3] * shared_transform[3 * BLOCK_SIZE + threadIdx.x];
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 4] * shared_transform[4 * BLOCK_SIZE + threadIdx.x];
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 5] * shared_transform[5 * BLOCK_SIZE + threadIdx.x];
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 6] * shared_transform[6 * BLOCK_SIZE + threadIdx.x];
    sums += shared_matrix[threadIdx.y * BLOCK_SIZE + 7] * shared_transform[7 * BLOCK_SIZE + threadIdx.x];

    result[Id_y * gridDim.x * BLOCK_SIZE + Id_x] = sums;
}

void dct_all_blocks_cuda(float* image_matrix, const int img_height, const int img_width, const float* transform_matrix, float* result)
{
    // Pre-alloca memoria GPU per i blocchi temporanei
    float *temp2,*d_Q_matrix;
    CHECK_CUDA(cudaMalloc(&temp2, img_width * img_height * sizeof(float)));
    CHECK_CUDA(cudaGetSymbolAddress((void**)&d_Q_matrix,const_quant_matrix));

    // Configurazione della griglia e dei blocchi
    // -> using BLOCK SIZE
    const int gridx = img_width / BLOCK_SIZE;
    const int gridy = img_height / BLOCK_SIZE;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(gridx, gridy);

    // Avvia il timer
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Lancio del kernel subsampling (--128)
    sub_matrix_scalar<<<gridDim, blockDim>>>(image_matrix, 128, image_matrix, img_width * img_height);

    // Lancio del kernel DCT
    cuda_matrix_dct<<<gridDim, blockDim>>>(image_matrix, transform_matrix, temp2);

    // Lancio del kernel quantizzazione
    divide_matrices<<<gridDim, blockDim>>>(temp2, d_Q_matrix, result, img_width * img_height);

    // Ferma il timer
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    // Calcola il tempo totale
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("DCT (%d,%d): %f ms\n",img_width,img_height,milliseconds);

    // Libera memoria GPU
    CHECK_CUDA(cudaFree(temp2));
}

void idct_all_blocks_cuda(const float* image_matrix, const int img_height, const int img_width, const float* transform_matrix, float* result)
{
    // Pre-alloca memoria GPU per i blocchi temporanei
    float* temp2, *d_Q_matrix;
    CHECK_CUDA(cudaMalloc(&temp2, img_width * img_height * sizeof(float)));
    CHECK_CUDA(cudaGetSymbolAddress((void**)&d_Q_matrix,const_quant_matrix));

    // Configurazione della griglia e dei blocchi
    // -> using BLOCK_SIZE
    const int gridx = img_width / BLOCK_SIZE;
    const int gridy = img_height / BLOCK_SIZE;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(gridx, gridy);

    // Avvia il timer
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Lancio del kernel de-quantizzazione
    multiply_matrices<<<gridDim, blockDim>>>(image_matrix, d_Q_matrix, temp2, img_width * img_height);

    // Lancio del kernel IDCT
    cuda_matrix_idct<<<gridDim, blockDim>>>(temp2, transform_matrix, result);

    // Lancio del kernel inverse of subsampling (++128)
    add_matrix_scalar<<<gridDim, blockDim>>>(result, 128, result, img_width * img_height);

    // Ferma il timer
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    // Calcola il tempo totale
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("IDCT (%d,%d): %f ms\n",img_width,img_height,milliseconds);

    // Libera memoria GPU
    CHECK_CUDA(cudaFree(temp2));
}