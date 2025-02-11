//%%writefile benchmark_fastAppr.cu
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

    // Cleanup
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
 * (TRANSFORM_MATRIX @ IMAGE) @ TRANSFORM_MATRIX.T
 * La matrice di trasformazione è 8x8
 * shared_matrix = TRANSFORM_MATRIX @ IMAGE
 * result = shared_matrix @ TRANSFORM_MATRIX.T
 * */
__global__ void cuda_matrix_dct_paper(const float* image_matrix, const int img_size, const float* transform_matrix, float* result) {
    float riga[BLOCK_SIZE];
    __shared__ float shared_transform[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float shared_image[BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE];
    /* *
     * shared_image logicamente divisa come un blocco 8x8 su una sola riga
     * Sono 8 blocchi 8x8, 8 righe da 64 float.
     * [[image_block_0]
     *  [image_block_i]
     *  [image_block_7]]
     * */
    // CUDA related vars (ids)
    int Id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int Id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global = Id_y * gridDim.x * blockDim.x + Id_x;
    // Image size related vars ()
    int imgDimX = img_size / blockDim.x; // numero di blocchi 8x8 sull asse X
    int imgDimY = img_size / blockDim.y; // numero di blocchi 8x8 sull'asse Y
    int imageIdY = global / BLOCK_SIZE / imgDimX; // indicizzazione del blocco, Y
    int imageIdX = global / BLOCK_SIZE % imgDimY; // indicizzazione del blocco, X
    int offset_y = imageIdY * BLOCK_SIZE * img_size; // si "sposta" verso il basso di (BLOCK_SIZE * img_size)
    int offset_x = imageIdX * BLOCK_SIZE; // si "sposta" verso destra di BLOCK_SIZE

    float sums = 0;

    shared_transform[threadIdx.y * blockDim.x + threadIdx.x] = transform_matrix[threadIdx.y * blockDim.x + threadIdx.x];
    for(int i = 0;i < BLOCK_SIZE;i++)
        shared_image[threadIdx.y*(BLOCK_SIZE*BLOCK_SIZE) + (threadIdx.x*BLOCK_SIZE) + i] = image_matrix[(offset_y+offset_x) + (threadIdx.x*img_size) + i];
        //shared_image[VA CAMBIATO] = image_matrix[(offset_y) + (threadIdx.y*img_size) + (threadIdx.x*BLOCK_SIZE) + i];
    __syncthreads();
    /* *
     * Il seguente IF serve per evitare accessi illegali alla memoria.
     * Essendo che adesso i Threads sono mappati sul blocco immagine,
     * potrebbero esserci thread che non svolgono lavoro.
     * In tal caso bisogna evitare che accedano alla memoria.
     * */
    if (global >= imgDimX * imgDimY * BLOCK_SIZE)return;

    // RIGHE DI T PER COLONNE DI IMG (T @ X)
    for (int i = 0;i < BLOCK_SIZE;i++) {
        for (int j = 0;j < BLOCK_SIZE;j++) {
            // sums += T [ sempre la stessa riga ] * X [ colonne in sequenza ]
            //sums += shared_transform[threadIdx.x * BLOCK_SIZE + j] * image_matrix[i + (offset_y + offset_x) + (j * img_size)];
            sums += shared_transform[threadIdx.x * BLOCK_SIZE + j] * shared_image[threadIdx.y*(BLOCK_SIZE*BLOCK_SIZE) + i + j*BLOCK_SIZE];
        }
        // TX [ riga ] = T[ riga ] * X[ colonne ] (TX[riga] = somma dei prodotti)
        // shared_matrix[(offset_y + offset_x) + threadIdx.x * img_size + i] = sums;
        riga[i] = sums;
        sums = 0;
    }

    sums = 0;

    // RIGHE DI TX PER RIGHE DI T (TX @ T.T)
    for (int i = 0;i < BLOCK_SIZE;i++) {
        for (int j = 0;j < BLOCK_SIZE;j++) {
            // sums += TX [ sempre la stessa riga ] * T [ righe in sequenza ]
            // sums += shared_matrix[(offset_y + offset_x) + threadIdx.x * img_size + j] * transform_matrix[i * 8 + j];
            sums+= riga[j] * shared_transform[i * BLOCK_SIZE + j];
        }
        // result [ riga ] = TX [ riga ] * T [ righe ]
        result[(offset_y + offset_x) + (threadIdx.x * img_size) + i] = sums;
        sums = 0;
    }
}

/* *
 * Effettua la IDCT utilizzando la matrice di trasformazione
 * (TRANSFORM_MATRIX.T @ DCT_MATRIX) @ TRANSFORM_MATRIX
 * La matrice di trasformazione è 8x8
 * shared_matrix = TRANSFORM_MATRIX.T @ DCT_MATRIX
 * result = shared_matrix @ TRANSFORM_MATRIX
 * */
__global__ void cuda_matrix_idct_paper(const float* image_matrix, const int img_size,const float* transform_matrix, float* result) {
    float riga[BLOCK_SIZE];
    __shared__ float shared_transform[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float shared_image[BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE];
    /* *
     * shared_image logicamente divisa come un blocco 8x8 su una sola riga
     * Sono 8 blocchi 8x8, 8 righe da 64 float.
     * [[image_block_0]
     *  [image_block_i]
     *  [image_block_7]]
     * */
    // CUDA related vars (ids)
    int Id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int Id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global = Id_y * gridDim.x * blockDim.x + Id_x;
    // Image size related vars ()
    int imgDimX = img_size / blockDim.x; // numero di blocchi 8x8 sull asse X
    int imgDimY = img_size / blockDim.y; // numero di blocchi 8x8 sull'asse Y
    int imageIdY = global / BLOCK_SIZE / imgDimX; // indicizzazione del blocco, Y
    int imageIdX = global / BLOCK_SIZE % imgDimY; // indicizzazione del blocco, X
    int offset_y = imageIdY * BLOCK_SIZE * img_size; // si "sposta" verso il basso di (BLOCK_SIZE * img_size)
    int offset_x = imageIdX * BLOCK_SIZE; // si "sposta" verso destra di BLOCK_SIZE

    float sums = 0;
    shared_transform[threadIdx.y * blockDim.x + threadIdx.x] = transform_matrix[threadIdx.y * blockDim.x + threadIdx.x];
    for(int i = 0;i < BLOCK_SIZE;i++)
        shared_image[threadIdx.y*(BLOCK_SIZE*BLOCK_SIZE) + (threadIdx.x*BLOCK_SIZE) + i] = image_matrix[(offset_y+offset_x) + (threadIdx.x*img_size) + i];
        //shared_image[VA CAMBIATO] = image_matrix[(offset_y) + (threadIdx.y*img_size) + (threadIdx.x*BLOCK_SIZE) + i];
    __syncthreads();
    /* *
     * Il seguente IF serve per evitare accessi illegali alla memoria.
     * Essendo che adesso i Threads sono mappati sul blocco immagine,
     * potrebbero esserci thread che non svolgono lavoro.
     * In tal caso bisogna evitare che accedano alla memoria.
     * */
    if (global >= imgDimX * imgDimY * BLOCK_SIZE)return;


    // COLONNE DI T PER COLONNE DI IMG (T.T @ X)
    for (int i = 0;i < BLOCK_SIZE;i++) {
        for (int j = 0;j < BLOCK_SIZE;j++) {
            // sums += TX [ sempre la stessa colonna ] * T [ colonne in sequenza ]
            //sums += shared_transform[threadIdx.x + j * BLOCK_SIZE] * image_matrix[i + (offset_y + offset_x) + (j * img_size)];
            sums += shared_transform[threadIdx.x + j * BLOCK_SIZE] * shared_image[threadIdx.y*(BLOCK_SIZE*BLOCK_SIZE) + i + j*BLOCK_SIZE];
        }
        // TX [ riga ] = T[ colonna ] * X[ colonne ] (TX[riga] = somma dei prodotti)
        // shared_matrix[(offset_y + offset_x) + threadIdx.x * img_size + i] = sums;
        riga[i]=sums;
        sums = 0;
    }

    sums = 0;

    // RIGHE DI TX PER COLONNE DI T (TX @ T)
    for (int i = 0;i < BLOCK_SIZE;i++) {
        for (int j = 0;j < BLOCK_SIZE;j++) {
            // sums += TX [ sempre la stessa riga ] * T [ colonne in sequenza ]
            // sums += shared_matrix[(offset_y + offset_x) + threadIdx.x * img_size + j] * transform_matrix[i + j * BLOCK_SIZE];
            sums += riga[j] * shared_transform[i + j * 8];
        }
        // result [ riga ] = TX [ riga ] * T [ colonne ]
        result[(offset_y + offset_x) + (threadIdx.x * img_size) + i] = sums;
        sums = 0;
    }
}

void dct_all_blocks_cuda(float* image_matrix, const int img_height, const int img_width, const float* transform_matrix, float* result)
{
    // Pre-alloca memoria GPU per i blocchi temporanei
    float* temp2;
    CHECK_CUDA(cudaMalloc(&temp2, img_width * img_height * sizeof(float)));

    // Configurazione della griglia e dei blocchi
    // -> using BLOCK SIZE
    int gridx = img_width / BLOCK_SIZE;
    int gridy = img_height / BLOCK_SIZE;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(gridx, gridy);
    int mono_grid_Dim = ((gridx * gridy * 8)+(BLOCK_SIZE*BLOCK_SIZE)-1)/(BLOCK_SIZE*BLOCK_SIZE);

    // Avvia il timer
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    // subsampling (--128)
    sub_matrix_scalar<<<gridDim, blockDim>>>(image_matrix, 128, image_matrix, img_width * img_height);

    // applica la DCT
    cuda_matrix_dct_paper<<<mono_grid_Dim, blockDim>>>(image_matrix, img_width, transform_matrix, temp2);

    // Applicazione della quantizzazione
    float q_matrix[BLOCK_SIZE * BLOCK_SIZE] = {
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99};

    // alloca quant_matrix on device
    float* d_Q_matrix;
    CHECK_CUDA(cudaMalloc(&d_Q_matrix, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_Q_matrix, q_matrix, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice));

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
    CHECK_CUDA(cudaFree(d_Q_matrix));
}

void idct_all_blocks_cuda(const float* image_matrix, const int img_height, const int img_width, const float* transform_matrix, float* result)
{
    // Pre-alloca memoria GPU per i blocchi temporanei
    float* temp2;
    CHECK_CUDA(cudaMalloc(&temp2, img_width * img_height * sizeof(float)));

    // Applicazione della de-quantizzazione
    float q_matrix[BLOCK_SIZE * BLOCK_SIZE] = {
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99 };

    // alloca quant_matrix on device
    float* d_Q_matrix;
    CHECK_CUDA(cudaMalloc(&d_Q_matrix, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_Q_matrix, q_matrix, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Configurazione della griglia e dei blocchi
    // -> using BLOCK_SIZE
    int gridx = img_width / BLOCK_SIZE;
    int gridy = img_height / BLOCK_SIZE;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(gridx, gridy);
    int mono_grid_Dim = ((gridx * gridy * 8)+(BLOCK_SIZE*BLOCK_SIZE)-1)/(BLOCK_SIZE*BLOCK_SIZE);

    // Avvia il timer
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Lancio del kernel de-quantizzazione
    multiply_matrices<<<gridDim, blockDim>>>(image_matrix, d_Q_matrix, temp2, img_width * img_height);

    // applica la IDCT
    cuda_matrix_idct_paper<<<mono_grid_Dim, blockDim>>>(temp2, img_width, transform_matrix, result);

    // inverse of subsampling (++128)
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
    CHECK_CUDA(cudaFree(d_Q_matrix));
}