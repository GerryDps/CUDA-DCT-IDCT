//%%cuda --compiler-args "--library cublas --library jpeg -arch=sm_75"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.cuh"
#include "utils_kernels.cuh"

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

__constant__ float const_quant_matrix[BLOCK_SIZE*BLOCK_SIZE];

// Using cuda kernels to compute the DCT and the IDCT
// Those FUNC use intenal dct/idct kernel function, differs according to implementation used.
void dct_all_blocks_cuda(float* image_matrix, const int img_height, const int img_width, const float* transform_matrix, float* result);
void idct_all_blocks_cuda(const float* image_matrix, const int img_height, const int img_width, const float* transform_matrix, float* result);

int main(int argc, char *argv[])
{
    if (argc != 3){
        fprintf(stderr, "Usage: %s <input_image> <output_image> \n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    const char *filename_out = argv[2];
    int width, height, channels;

    // Load a jpeg image in image_matrix
    unsigned char *image_matrix = load_jpeg_as_matrix(filename, &width, &height, &channels);
    if (!image_matrix)
    {
        exit(EXIT_FAILURE);
    }

    // allocate host memory for the float image
    float *image_matrix_float;
    image_matrix_float = (float *)malloc(width * height * sizeof(float));
    convertToFloat(image_matrix, image_matrix_float, width * height * channels);
    free(image_matrix);

    printf("Printing the 8x8 of image[] (matrix from the jpeg image w:%d h:%d)\n",width,height);
    for (int i = 0; i < BLOCK_SIZE; i++){
        for (int j = 0; j < BLOCK_SIZE; j++){
            printf("%f ", image_matrix_float[i * width + j]);
        }
        printf("\n");
    }
    printf("\n\n");

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

    printf("Printing the 8x8 of result[] (matrix coming from the dct)\n");
    for (int i = 0; i < BLOCK_SIZE; i++){
        for (int j = 0; j < BLOCK_SIZE; j++){
            printf("%f ", result[i * width + j]);
        }
        printf("\n");
    }
    printf("\n\n");

    // d_E = result of the idct applied on the block_image compressed
    float *d_E;
    CHECK_CUDA(cudaMalloc(&d_E, width * height * sizeof(float)));

    // Compute the idct
    // d_C = result ; d_B = transform_matrix ; d_E = result
    idct_all_blocks_cuda(d_C, height, width, d_B,d_E);

    // copy device memory to host
    // result = d_E
    CHECK_CUDA(cudaMemcpy(result, d_E, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Printing the 8x8 of result[] (matrix coming from the idct)\n");
    for (int i = 0; i < BLOCK_SIZE; i++){
        for (int j = 0; j < BLOCK_SIZE; j++){
            printf("%f ", result[i * width + j]);
        }
        printf("\n");
    }
    printf("\n\n");

    // Salva l'immagine in formato JPEG
    int quality = 100; // Qualita JPEG (0-100)

    // allocate host memory for the usigned char image
    unsigned char *image_matrix_uc;
    image_matrix_uc = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    convertToUnsignedChar(result, image_matrix_uc, width * height);
    free(result);

    printf("Printing the 8x8 of U_C[] (unsignedchar)\n");
    for (int i = 0; i < BLOCK_SIZE; i++){
        for (int j = 0; j < BLOCK_SIZE; j++){
            printf("%d ", image_matrix_uc[i * width + j]);
        }
        printf("\n");
    }
    printf("\n\n");

    if (save_grayscale_jpeg(filename_out, image_matrix_uc, width, height, quality))
    {
        printf("Image saved successfully to %s\n", filename_out);
    }
    else
    {
        fprintf(stderr, "Error: Failed to save image\n");
    }
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_E));
    free(image_matrix_uc);
    free(image_matrix_float);
    return 0;
}

/* *
 * Effettua la DCT utilizzando la matrice di trasformazione
 * (TRASFORM_MATRIX @ IMAGE_MATRIX) @ TRANSFORM_MATRIX.T
 * La matrice di trasformazione e 8x8
 * result = TRASFORM_MATRIX @ IMAGE_MATRIX
 * result = result @ TRANSFORM_MATRIX.T
 * */
__global__ void cuda_matrix_dct(const float* image_matrix, const float* transform_matrix, float* result) {
    __shared__ float shared_matrix[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float shared_transform[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float shared_image[BLOCK_SIZE*BLOCK_SIZE];
    // Calcola l'indice globale del thread
    const int Id_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int Id_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int global = Id_y * gridDim.x * BLOCK_SIZE + Id_x;

    float sums = 0;
    shared_transform[threadIdx.y * BLOCK_SIZE + threadIdx.x] = transform_matrix[threadIdx.y * BLOCK_SIZE + threadIdx.x];
    shared_image[threadIdx.y * BLOCK_SIZE + threadIdx.x] = image_matrix[global];
    __syncthreads();

    // result = transform_matrix @ image_matrix
    // result = transform_matrix[righe] @ image_matrix[colonne]
    for (int i = 0;i < BLOCK_SIZE;i++) {
        //sums += transform_matrix[threadIdx.y * BLOCK_SIZE + i] * image_matrix[(offset_y * gridDim.x * BLOCK_SIZE) + i * (gridDim.x * BLOCK_SIZE) + Id_x];
        //sums += shared_transform[threadIdx.y * BLOCK_SIZE + i] * image_matrix[(offset_y * gridDim.x * BLOCK_SIZE) + i * (gridDim.x * BLOCK_SIZE) + Id_x];
        sums += shared_transform[threadIdx.y * BLOCK_SIZE + i] * shared_image[threadIdx.x + i * BLOCK_SIZE];
    }
    // result[Id_y * gridDim.x * BLOCK_SIZE + Id_x] = sums;
    shared_matrix[threadIdx.y * BLOCK_SIZE + threadIdx.x] = sums;
    sums = 0;
    // Devo attendere il completamento della DOT precedente
    __syncthreads();

    // result = result(precedente) @ transform_matrix.T (trasposta)
    // result = result(precedente)[righe] @ transform_matrix[righe] (perche la trasposta)
    for (int i = 0;i < BLOCK_SIZE;i++) {
        //sums += result[Id_y * (gridDim.x * BLOCK_SIZE) + offset_x + i] * transform_matrix[threadIdx.x * BLOCK_SIZE + i];
        sums += shared_matrix[threadIdx.y * BLOCK_SIZE + i] * shared_transform[threadIdx.x * BLOCK_SIZE + i];
    }
    result[Id_y * gridDim.x * BLOCK_SIZE + Id_x] = sums;
}

/* *
 * Effettua la IDCT utilizzando la matrice di trasformazione
 * (TRANSFORM_MATRIX.T @ DCT_MATRIX) @ TRANSFORM_MATRIX
 * La matrice di trasformazione e 8x8
 * result = TRANSFORM_MATRIX.T @ DCT_MATRIX
 * result = result @ TRANSFORM_MATRIX
 * */
__global__ void cuda_matrix_idct(const float* image_matrix, const float* transform_matrix, float* result) {
    __shared__ float shared_matrix[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float shared_transform[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float shared_image[BLOCK_SIZE*BLOCK_SIZE];
    // Calcola l'indice globale del thread
    const int Id_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int Id_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int global = Id_y * gridDim.x * BLOCK_SIZE + Id_x;

    float sums = 0;
    shared_transform[threadIdx.y * BLOCK_SIZE + threadIdx.x] = transform_matrix[threadIdx.y * BLOCK_SIZE + threadIdx.x];
    shared_image[threadIdx.y * BLOCK_SIZE + threadIdx.x] = image_matrix[global];
    __syncthreads();

    // result = transform_matrix.T @ dct_matrix
    // result = transform_matrix[colonne](x trasposta) @ image_matrix[colonne]
    for (int i = 0;i < BLOCK_SIZE;i++) {
        //sums += shared_transform[i * BLOCK_SIZE + threadIdx.y] * image_matrix[(offset_y * gridDim.x * BLOCK_SIZE) + i * (gridDim.x * BLOCK_SIZE) + Id_x];
        sums += shared_transform[i * BLOCK_SIZE + threadIdx.y] * shared_image[threadIdx.x + i * BLOCK_SIZE];
    }
    shared_matrix[threadIdx.y * BLOCK_SIZE + threadIdx.x] = sums;
    sums = 0;
    __syncthreads();

    // result = result(precedente) @ transform_matrix
    // result = result[righe] @ transform_matrix[colonne]
    for (int i = 0;i < BLOCK_SIZE;i++) {
        sums += shared_matrix[threadIdx.y * BLOCK_SIZE + i] * shared_transform[i * BLOCK_SIZE + threadIdx.x];
    }
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