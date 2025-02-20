//%%writefile benchmark_cublass_2.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

// Check CUBLAS error
#define CHECK_CUBLAS(call)                                         \
    {                                                              \
        cublasStatus_t err = call;                                 \
        if (err != CUBLAS_STATUS_SUCCESS)                          \
        {                                                          \
            printf("CUBLAS error in %s : %d", __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

#define BLOCK_SIZE 8

__constant__ float const_quant_matrix[BLOCK_SIZE*BLOCK_SIZE];

// Kernels CUDA per le operazioni aritmetiche element-wise
__global__ void sub_matrix_scalar(const float* A, const float scalar, float* C, int size);
__global__ void add_matrix_scalar(const float* A, const float scalar, float* C, int size);

__global__ void divide_matrices(const float* A, const float* B, float* C, int size);
__global__ void multiply_matrices(const float* A, const float* B, float* C, int size);

// Using CUBLAS HANDLE to compute the DCT and the IDCT
__host__ void dct_all_blocks(float *image_matrix, int img_height, int img_width, const float *transform_matrix, float *result, cublasHandle_t handle);
__host__ void idct_all_blocks(float *image_matrix, int img_height, int img_width, const float *transform_matrix, float *result, cublasHandle_t handle);

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

    // Initialize CUBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

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

    // Compute DCT using CUBLAS
    // d_A = image_block | d_B = transform_matrix | d_C = result
    dct_all_blocks(d_A, height,width,d_B, d_C, handle);

    // copy result form device memory back to host
    // result = d_C
    CHECK_CUDA(cudaMemcpy(result, d_C, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    // d_E = result of the idct applied on the block_image compressed
    float *d_E;
    CHECK_CUDA(cudaMalloc(&d_E, width * height * sizeof(float)));

    // Compute the idct
    // d_C = result ; d_B = transform_matrix ; d_E = result
    idct_all_blocks(d_C, height, width, d_B,d_E,handle);

    // copy device memory to host
    // result = d_E
    CHECK_CUDA(cudaMemcpy(result, d_E, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
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


/* *
 * Funzione per l'applicazione della "compressione" (IN_IMAGE->scaling->dct->quantization->OUT_COMPRESSED)
 * L'utilizzo di una matrice di quantizzazione con tutti 1 ad esempio, ovviamente non porta a nessuna compressione
 * e/o perdita di qualità, in quando dct/idct sono reversibili a meno di piccoli errori di arrotondamento.
 * In questo caso particolare la matrice di quantizzazione è quella definita da JPEG (diversa da 1),
 * e quindi introduce compressione/perdita di qualità.
 * */
void dct_all_blocks(float *image_matrix, int img_height, int img_width, const float *transform_matrix, float *result, cublasHandle_t handle)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    // Pre-alloca memoria GPU per i blocchi temporanei
    float *temp1, *transform_matrix_expanded, *d_Q_matrix;
    CHECK_CUDA(cudaMalloc(&temp1, img_width * img_height * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&transform_matrix_expanded, img_width * img_height * sizeof(float)));
    CHECK_CUDA(cudaGetSymbolAddress((void**)&d_Q_matrix,const_quant_matrix));

    // Configurazione della griglia e dei blocchi
    // -> using BLOCK_SIZE
    int gridx = img_width / BLOCK_SIZE;
    int gridy = img_width / BLOCK_SIZE;
    dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE);
    dim3 gridDim(gridx,gridy);

    // Avvia il timer
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    // subsampling (--128)
    sub_matrix_scalar<<<gridDim,blockDim>>>(image_matrix, 128, image_matrix, img_width * img_height);

    // expand the T matrix to the same size of the image
    upgrade_T_matrix<<<gridDim,blockDim>>>(transform_matrix,transform_matrix_expanded,img_width * img_height);

    // Calcola temp1 = transform_matrix_expanded @ image_block
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, img_height, img_width, img_width,
                             &alpha, transform_matrix_expanded, img_height, image_matrix, img_width,
                             &beta, temp1, img_height));

    // Calcola temp1 @ transform_matrix_expanded.T
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, img_height, img_width, img_width,
                             &alpha, temp1, img_height, transform_matrix_expanded, img_width,
                             &beta, result, img_height));

    // Lancio del kernel quantizzazione
    divide_matrices<<<gridDim,blockDim>>>(result, d_Q_matrix, result, img_width * img_height);

    // Ferma il timer
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    // Calcola il tempo totale
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("DCT (%d,%d): %f ms\n",img_width,img_height,milliseconds);


    // Libera memoria GPU
    CHECK_CUDA(cudaFree(temp1));
    CHECK_CUDA(cudaFree(transform_matrix_expanded));
}

/* *
 * Funzione per l'applicazione della "de-compressione" (IN_COMPRESSED->de-quantization->idct->UPscaling->OUT_IMAGE)
 * */
void idct_all_blocks(float *image_matrix, int img_height, int img_width, const float *transform_matrix, float *result, cublasHandle_t handle)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    // Pre-alloca memoria GPU per i blocchi temporanei
    float *temp1, *transform_matrix_expanded, *d_Q_matrix;
    CHECK_CUDA(cudaMalloc(&temp1, img_width * img_height * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&transform_matrix_expanded, img_width * img_height * sizeof(float)));
    CHECK_CUDA(cudaGetSymbolAddress((void**)&d_Q_matrix,const_quant_matrix));

    // Configurazione della griglia e dei blocchi
    // -> using BLOCK_SIZE
    int gridx = img_width / BLOCK_SIZE;
    int gridy = img_width / BLOCK_SIZE;
    dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE);
    dim3 gridDim(gridx,gridy);

    // Avvia il timer
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Lancio del kernel de-quantizzazione
    multiply_matrices<<<gridDim,blockDim>>>(image_matrix, d_Q_matrix, image_matrix, img_width * img_height);

    // expand the T matrix to the same size of the image
    upgrade_T_matrix<<<gridDim,blockDim>>>(transform_matrix,transform_matrix_expanded,img_width * img_height);

    // Compute temp1 = transform_matrix.T @ image_block
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, img_height, img_width, img_width,
                             &alpha, transform_matrix_expanded, img_height, image_matrix, img_width,
                             &beta, temp1, img_height));

    // Compute temp1 @ transform_matrix
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, img_height, img_width, img_width,
                             &alpha, temp1, img_height, transform_matrix_expanded, img_width,
                             &beta, result, img_height));

    // inverse of subsampling (++128)
    add_matrix_scalar<<<gridDim,blockDim>>>(result, 128, result, img_width * img_height);

    // Ferma il timer
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    // Calcola il tempo totale
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("IDCT (%d,%d): %f ms\n",img_width,img_height,milliseconds);

    // Libera memoria GPU
    CHECK_CUDA(cudaFree(temp1));
    CHECK_CUDA(cudaFree(transform_matrix_expanded));
}
