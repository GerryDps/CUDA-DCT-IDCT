#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <jpeglib.h>
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

// Host function to load image as matrix
unsigned char *
load_jpeg_as_matrix(const char *filename, int *width, int *height, int *channels);

// Host function to compute DCT using CUBLAS
void dct_block(const float *image_block, const float *transform_matrix, float *result, cublasHandle_t handle);

// Host function to compute IDCT using CUBLAS
void idct_block(const float *image_block, const float *transform_matrix, float *result, cublasHandle_t handle);

void dct_all_blocks(const float *image_matrix, int img_height, int img_width, const float *transform_matrix, float *result, cublasHandle_t handle);

int main()
{

    /*const char *filename = "filename.jpg";
    int width, height, channels;

    // Load a jpeg image in image_matrix
    unsigned char *image_matrix = load_jpeg_as_matrix(filename, &width, &height, &channels);
    if (!image_matrix)
    {
        exit(EXIT_FAILURE);
    }*/

    // image_matrix
    float image_matrix[BLOCK_SIZE * BLOCK_SIZE] = {
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99};

    // Quantization matrix
    float quant_matrix[BLOCK_SIZE * BLOCK_SIZE] = {
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99};

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

    // allocate host memory for result
    float *result;
    result = (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float));

    // allocate device memory for: image_block, trasform_matrix and result
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)));

    // copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_A, image_matrix, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, transform_matrix, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Compute DCT using CUBLAS
    // d_A = image_block ; d_B = transform_matrix ; d_C = result
    //dct_block(d_A, d_B, d_C, handle);
    dct_all_blocks(d_A, BLOCK_SIZE,BLOCK_SIZE,d_B, d_C, handle);

    // copy device memory to host
    // result = d_C
    CHECK_CUDA(cudaMemcpy(result, d_C, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < BLOCK_SIZE; i++)
        for (int j = 0; j < BLOCK_SIZE; j++)
            printf("%f ", result[i * BLOCK_SIZE + j]);

    // d_E = rsult of idct of the blockimage compressed
    float *d_E;
    CHECK_CUDA(cudaMalloc(&d_E, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)));

    // Compute the idct
    // d_C = result ; d_B = transform_matrix ; d_E = result
    idct_block(d_C, d_B, d_E, handle);

    // copy device memory to host
    // result = d_E
    CHECK_CUDA(cudaMemcpy(result, d_E, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < BLOCK_SIZE; i++)
        for (int j = 0; j < BLOCK_SIZE; j++)
            printf("%f ", result[i * BLOCK_SIZE + j]);

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}

// Host function to load image as matrix
unsigned char *load_jpeg_as_matrix(const char *filename, int *width, int *height, int *channels)
{
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE *infile = fopen(filename, "rb");
    if (!infile)
    {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        return NULL;
    }

    // Set up error handling
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    // Specify the source of the data (the input file)
    jpeg_stdio_src(&cinfo, infile);

    // Read the JPEG header to get image info
    jpeg_read_header(&cinfo, TRUE);

    // Start decompression
    jpeg_start_decompress(&cinfo);

    *width = cinfo.output_width;
    *height = cinfo.output_height;
    *channels = cinfo.output_components; // 1 for grayscale, 3 for RGB
    int colorspace = cinfo.out_color_space;
    printf("Color Space: %d\n", colorspace);

    // Allocate memory for the pixel matrix
    unsigned long matrix_size = (*width) * (*height) * (*channels);
    unsigned char *image_matrix = (unsigned char *)malloc(matrix_size);
    if (!image_matrix)
    {
        fprintf(stderr, "Error: Unable to allocate memory for image matrix\n");
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return NULL;
    }

    // Read scanlines into the matrix row by row
    unsigned char *row_pointer[1];
    while (cinfo.output_scanline < cinfo.output_height)
    {
        row_pointer[0] = image_matrix + (cinfo.output_scanline * (*width) * (*channels));
        jpeg_read_scanlines(&cinfo, row_pointer, 1);
    }

    // Finish decompression and clean up
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return image_matrix;
}

// Host function to compute DCT using CUBLAS
void dct_block(const float *image_block, const float *transform_matrix, float *result, cublasHandle_t handle)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    float *temp; // temp = transform_matrix @ image_block
    CHECK_CUDA(cudaMalloc(&temp, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)));

    // Compute transform_matrix @ image_block
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, &alpha, transform_matrix, BLOCK_SIZE, image_block, BLOCK_SIZE, &beta, temp, BLOCK_SIZE));

    // Compute temp @ transform_matrix.T // (temp = transform_matrix @ image_block)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, &alpha, temp, BLOCK_SIZE, transform_matrix, BLOCK_SIZE, &beta, result, BLOCK_SIZE));

    CHECK_CUDA(cudaFree(temp));
}

// Host function to compute IDCT using CUBLAS
void idct_block(const float *image_block, const float *transform_matrix, float *result, cublasHandle_t handle)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    float *temp; // temp = transform_matrix.T @ image_block
    CHECK_CUDA(cudaMalloc(&temp, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)));

    // Compute transform_matrix.T @ image_block
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, &alpha, transform_matrix, BLOCK_SIZE, image_block, BLOCK_SIZE, &beta, temp, BLOCK_SIZE));

    // Compute (transform_matrix.T @ image_block) @ transform_matrix
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, &alpha, temp, BLOCK_SIZE, transform_matrix, BLOCK_SIZE, &beta, result, BLOCK_SIZE));

    CHECK_CUDA(cudaFree(temp));
}

void dct_all_blocks(const float *image_matrix, int img_height, int img_width, const float *transform_matrix, float *result, cublasHandle_t handle)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    // Pre-alloca memoria GPU per i blocchi temporanei
    float *temp1;
    CHECK_CUDA(cudaMalloc(&temp1, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)));

    // Itera sui blocchi 8x8
    for (int block_row = 0; block_row < img_height; block_row += BLOCK_SIZE)
    {
        for (int block_col = 0; block_col < img_width; block_col += BLOCK_SIZE)
        {

            // Calcola l'offset del blocco corrente
            const float *image_block = image_matrix + block_row * img_width + block_col;
            float *result_block = result + block_row * img_width + block_col;

            // Calcola transform_matrix @ image_block
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE,
                                     &alpha, transform_matrix, BLOCK_SIZE, image_block, img_width,
                                     &beta, temp1, BLOCK_SIZE));

            // Calcola temp1 @ transform_matrix.T
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE,
                                     &alpha, temp1, BLOCK_SIZE, transform_matrix, BLOCK_SIZE,
                                     &beta, result_block, img_width));

        }
    }

    // Libera memoria GPU
    CHECK_CUDA(cudaFree(temp1));
    CHECK_CUDA(cudaFree(temp2));
}