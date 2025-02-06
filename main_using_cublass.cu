//%%cuda --compiler-args "--library cublas --library jpeg -arch=sm_75"
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
#define IMAGE_SIZE 256

__constant__ float const_quant_matrix[BLOCK_SIZE*BLOCK_SIZE];

// Kernels CUDA per le operazioni aritmetiche element-wise
__global__ void sub_matrix_scalar(const float* A, const float scalar, float* C, int size);
__global__ void add_matrix_scalar(const float* A, const float scalar, float* C, int size);

__global__ void divide_matrices(const float* A, const float* B, float* C, int size);
__global__ void multiply_matrices(const float* A, const float* B, float* C, int size);

// Host function to load image as matrix
static unsigned char *load_jpeg_as_matrix(const char *filename, int *width, int *height, int *channels);

// Host function to save matrix as jpeg
int save_grayscale_jpeg(const char *filename, unsigned char *image_matrix, int width, int height, int quality);

// Utils
void convertToFloat(unsigned char *input, float *output, int size);
void convertToUnsignedChar(const float *image_float, unsigned char *image_char, int size);

// Using CUBLAS HANDLE to compute the DCT and the IDCT
__host__ void dct_all_blocks(float *image_matrix, int img_height, int img_width, const float *transform_matrix, float *result, cublasHandle_t handle);
__host__ void idct_all_blocks(const float *image_matrix, int img_height, int img_width, const float *transform_matrix, float *result, cublasHandle_t handle);

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
    int quality = 100; // Qualità JPEG (0-100)

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

// Host function to load image as matrix
static unsigned char *load_jpeg_as_matrix(const char *filename, int *width, int *height, int *channels)
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

int save_grayscale_jpeg(const char *filename, unsigned char *image_matrix, int width, int height, int quality)
{
    // Strutture di compressione JPEG
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    // Imposta il gestore degli errori
    cinfo.err = jpeg_std_error(&jerr);

    // Inizializza l'oggetto di compressione JPEG
    jpeg_create_compress(&cinfo);

    // Apri il file in scrittura
    FILE *outfile = fopen(filename, "wb");
    if (!outfile)
    {
        fprintf(stderr, "Error: Unable to open file %s for writing\n", filename);
        return 0;
    }

    // Associa il file di output al compressore
    jpeg_stdio_dest(&cinfo, outfile);

    // Imposta i parametri dell'immagine JPEG
    cinfo.image_width = width;            // Larghezza in pixel
    cinfo.image_height = height;          // Altezza in pixel
    cinfo.input_components = 1;           // Numero di canali (1 per scala di grigi)
    cinfo.in_color_space = JCS_GRAYSCALE; // Colore: scala di grigi

    // Imposta i parametri di default e modifica la qualità
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    // Inizia la compressione
    jpeg_start_compress(&cinfo, TRUE);

    // Scrive ogni riga dell'immagine
    while (cinfo.next_scanline < cinfo.image_height)
    {
        unsigned char *row_pointer = &image_matrix[cinfo.next_scanline * width];
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    // Termina la compressione
    jpeg_finish_compress(&cinfo);

    // Libera le risorse
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);

    return 1;
}

// Convert an unsigned char image matrix to an float image matrix
void convertToFloat(unsigned char *input, float *output, int size)
{
    for (int i = 0; i < size; i++) {
        output[i] = (float)input[i];
    }
}

// Convert a float image matrix to unsigned char image matrix
void convertToUnsignedChar(const float *image_float, unsigned char *image_char, int size) {
    for (int i = 0; i < size; i++) {
        image_char[i] = (unsigned char)fminf(fmaxf(image_float[i], 0.0f), 255.0f); // Clamp tra 0 e 255
        //image_char[i] = (unsigned char)image_float[i]; // Clamp tra 0 e 255
    }
}

// Kernel CUDA per la sottrazione element-wise matrice - scalare
__global__ void sub_matrix_scalar(const float* A, const float scalar, float* C, int size) {
    // Calcola l'indice globale del thread
    int Id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int Id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global = Id_y * gridDim.x * blockDim.x + Id_x;

    // Controlla che l'indice sia all'interno dei limiti
    if (global < size) {
        C[global] =  A[global] - scalar;
    }
}

// Kernel CUDA per l'addizione element-wise matrice - scalare
__global__ void add_matrix_scalar(const float* A, const float scalar, float* C, int size) {
    // Calcola l'indice globale del thread
    int Id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int Id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global = Id_y * gridDim.x * blockDim.x + Id_x;

    // Controlla che l'indice sia all'interno dei limiti
    if (global < size) {
        C[global] =  A[global] + scalar;
    }
}

// Kernel CUDA per la divisione elemento per elemento
__global__ void divide_matrices(const float* A, const float* B, float* C, int size) {
    // Calcola l'indice globale del thread
    int Id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int Id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global = Id_y * gridDim.x * blockDim.x + Id_x;

    // Controlla che l'indice sia all'interno dei limiti
    if (global < size) {
        C[global] =  round(A[global] / B[threadIdx.x * blockDim.x + threadIdx.y]);
    }
}

// Kernel CUDA per la moltiplicazione elemento per elemento
__global__ void multiply_matrices(const float* A, const float* B, float* C, int size) {
    // Calcola l'indice globale del thread
    int Id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int Id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global = Id_y * gridDim.x * blockDim.x + Id_x;

    // Controlla che l'indice sia all'interno dei limiti
    if (global < size) {
        C[global] =  A[global] * B[threadIdx.x * blockDim.x + threadIdx.y];
    }
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
    float *temp1, *temp2,*d_Q_matrix;
    CHECK_CUDA(cudaMalloc(&temp1, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&temp2, img_width * img_height * sizeof(float)));
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

    // Itera sui blocchi 8x8 - applica la DCT
    for (int block_row = 0; block_row < img_height; block_row += BLOCK_SIZE)
    {
        for (int block_col = 0; block_col < img_width; block_col += BLOCK_SIZE)
        {
            // Calcola l'offset del blocco corrente
            const float *image_block = image_matrix + block_row * img_width + block_col;
            float *result_block = temp2 + block_row * img_width + block_col;

            // Calcola temp1 = transform_matrix @ image_block
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE,
                                     &alpha, transform_matrix, BLOCK_SIZE, image_block, img_width,
                                     &beta, temp1, BLOCK_SIZE));

            // Calcola temp1 @ transform_matrix.T
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE,
                                     &alpha, temp1, BLOCK_SIZE, transform_matrix, BLOCK_SIZE,
                                     &beta, result_block, img_width));
        }
    }

    // Lancio del kernel quantizzazione
    divide_matrices<<<gridDim,blockDim>>>(temp2, d_Q_matrix, result, img_width * img_height);

    // Ferma il timer
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    // Calcola il tempo totale
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("DCT (%d,%d): %f ms\n",img_width,img_height,milliseconds);


    // Libera memoria GPU
    CHECK_CUDA(cudaFree(temp1));
    CHECK_CUDA(cudaFree(temp2));
}

/* *
 * Funzione per l'applicazione della "de-compressione" (IN_COMPRESSED->de-quantization->idct->UPscaling->OUT_IMAGE)
 * */
void idct_all_blocks(const float *image_matrix, int img_height, int img_width, const float *transform_matrix, float *result, cublasHandle_t handle)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    // Pre-alloca memoria GPU per i blocchi temporanei
    float *temp1, *temp2,*d_Q_matrix;;
    CHECK_CUDA(cudaMalloc(&temp1, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&temp2, img_width * img_height * sizeof(float)));
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
    multiply_matrices<<<gridDim,blockDim>>>(image_matrix, d_Q_matrix, temp2, img_width * img_height);

    // Itera sui blocchi 8x8 - applica la IDCT
    for (int block_row = 0; block_row < img_height; block_row += BLOCK_SIZE)
    {
        for (int block_col = 0; block_col < img_width; block_col += BLOCK_SIZE)
        {
            // Calcola l'offset del blocco corrente
            const float *image_block = temp2 + block_row * img_width + block_col;
            float *result_block = result + block_row * img_width + block_col;

            // Compute temp1 = transform_matrix.T @ image_block
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE,
                                     &alpha, transform_matrix, BLOCK_SIZE, image_block, img_width,
                                     &beta, temp1, BLOCK_SIZE));

            // Compute temp1 @ transform_matrix
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE,
                                     &alpha, temp1, BLOCK_SIZE, transform_matrix, BLOCK_SIZE,
                                     &beta, result_block, img_width));
        }
    }

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
    CHECK_CUDA(cudaFree(temp2));
}
