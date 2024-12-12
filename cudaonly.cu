#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 8

// Funzione DCT per un blocco 8x8
__device__ void dct(float block[BLOCK_SIZE][BLOCK_SIZE], float result[BLOCK_SIZE][BLOCK_SIZE])
{
    for (int u = 0; u < BLOCK_SIZE; u++)
    {
        for (int v = 0; v < BLOCK_SIZE; v++)
        {
            float sum = 0.0f;
            for (int x = 0; x < BLOCK_SIZE; x++)
            {
                for (int y = 0; y < BLOCK_SIZE; y++)
                {
                    float alpha_u = (u == 0) ? sqrtf(1.0f / BLOCK_SIZE) : sqrtf(2.0f / BLOCK_SIZE);
                    float alpha_v = (v == 0) ? sqrtf(1.0f / BLOCK_SIZE) : sqrtf(2.0f / BLOCK_SIZE);

                    sum += block[x][y] * cosf((2 * x + 1) * u * M_PI / (2 * BLOCK_SIZE)) *
                           cosf((2 * y + 1) * v * M_PI / (2 * BLOCK_SIZE)) * alpha_u * alpha_v;
                }
            }
            result[u][v] = sum;
        }
    }
}

// Kernel CUDA per elaborare i blocchi 8x8
__global__ void process_blocks(float *image_matrix, float *image_result, int width, int height)
{
    // Identificatore del blocco nella griglia
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;

    // Identificatore del pixel all'interno del blocco
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    // Coordinate assolute del pixel nella matrice
    int x = blockX * BLOCK_SIZE + threadX;
    int y = blockY * BLOCK_SIZE + threadY;

    // Shared memory per il blocco corrente
    __shared__ float image_block[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float result_block[BLOCK_SIZE][BLOCK_SIZE];

    // Copia i dati nella shared memory
    if (x < width && y < height)
    {
        image_block[threadY][threadX] = image_matrix[y * width + x];
    }
    else
    {
        image_block[threadY][threadX] = 0.0f; // Padding per bordi
    }

    __syncthreads(); // Sincronizza i thread per assicurarsi che il blocco sia caricato

    // Applicazione della DCT al blocco
    if (threadX == 0 && threadY == 0)
    {
        dct(image_block, result_block);
    }

    __syncthreads(); // Sincronizza prima di scrivere i risultati

    // Scrive i risultati nella matrice di output
    if (x < width && y < height)
    {
        image_result[y * width + x] = result_block[threadY][threadX];
    }
}

int main()
{
    int width = 1024;  // Larghezza dell'immagine
    int height = 1024; // Altezza dell'immagine

    size_t size = width * height * sizeof(float);

    // Allocazione della memoria sull'host
    float *h_image_matrix = (float *)malloc(size);
    float *h_image_result = (float *)malloc(size);

    // Inizializza i valori dell'immagine
    for (int i = 0; i < width * height; i++)
    {
        h_image_matrix[i] = rand() % 256;
    }

    // Allocazione della memoria sulla GPU
    float *d_image_matrix, *d_image_result;
    cudaMalloc((void **)&d_image_matrix, size);
    cudaMalloc((void **)&d_image_result, size);

    // Copia i dati dall'host alla GPU
    cudaMemcpy(d_image_matrix, h_image_matrix, size, cudaMemcpyHostToDevice);

    // Configura i parametri della griglia e dei blocchi
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Lancia il kernel
    process_blocks<<<numBlocks, threadsPerBlock>>>(d_image_matrix, d_image_result, width, height);

    // Copia i risultati dalla GPU all'host
    cudaMemcpy(h_image_result, d_image_result, size, cudaMemcpyDeviceToHost);

    // Libera la memoria sulla GPU
    cudaFree(d_image_matrix);
    cudaFree(d_image_result);

    // Libera la memoria sull'host
    free(h_image_matrix);
    free(h_image_result);

    return 0;
}