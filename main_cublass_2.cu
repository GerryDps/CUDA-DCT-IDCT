//%%cuda --compiler-args "--library cublas --library jpeg -arch=sm_75"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.cuh"
#include "utils_kernels.cuh"
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

// Using CUBLAS HANDLE to compute the DCT and the IDCT
__host__ void dct_all_blocks(float *image_matrix, int img_height, int img_width, const float *transform_matrix, float *result, cublasHandle_t handle);
__host__ void idct_all_blocks(float *image_matrix, int img_height, int img_width, const float *transform_matrix, float *result, cublasHandle_t handle);

int main()
{

    const char *filename = "camera256.tif.jpeg";
    int width, height, channels;

    /*// Load a jpeg image in image_matrix
    unsigned char *image_matrix = load_jpeg_as_matrix(filename, &width, &height, &channels);
    if (!image_matrix)
    {
        exit(EXIT_FAILURE);
    }

    // allocate host memory for the float image
    float *image_matrix_float;
    image_matrix_float = (float *)malloc(width * height * sizeof(float));
    convertToFloat(image_matrix, image_matrix_float, width * height * channels);
    free(image_matrix);*/

    width = 4096;
    height = 4096;

    float* image_matrix_float;
    image_matrix_float = (float*)malloc(width * height * sizeof(float));
    srand(41);
    for (int i = 0; i < height;i++) {
        for (int j = 0; j < width; j++) {
            image_matrix_float[i * width + j] = rand() % 256;;
        }
    }

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
    idct_all_blocks(d_C, height, width, d_B,d_E,handle);

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
    const char *filename_out = "output.jpg";
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


    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_E));
    free(image_matrix_uc);
    free(image_matrix_float);
    return 0;
}

/* *
 * Funzione per l'applicazione della "compressione" (IN_IMAGE->scaling->dct->quantization->OUT_COMPRESSED)
 * L'utilizzo di una matrice di quantizzazione con tutti 1 ad esempio, ovviamente non porta a nessuna compressione
 * e/o perdita di qualita, in quando dct/idct sono reversibili a meno di piccoli errori di arrotondamento.
 * In questo caso particolare la matrice di quantizzazione e quella definita da JPEG (diversa da 1),
 * e quindi introduce compressione/perdita di qualita.
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
