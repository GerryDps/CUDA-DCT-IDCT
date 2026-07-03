# A GPU Accelerated DCT Implementation for Image Compression

[Cardone A.](mailto:ancardone@unisa.it), [Di Pascale G.](mailto:dipascalegerardo@gmail.com) (2026).

A GPU Accelerated DCT Implementation for Image Compression. In: Gervasi, O., et al. Computational Science and Its Applications – ICCSA 2026 Workshops. ICCSA 2026. Lecture Notes in Computer Science, vol 16763. Springer, Cham. https://doi.org/10.1007/978-3-032-30530-5_12

---

This repository contains the parallel and GPU-optimized implementation of the Discrete Cosine Transform (DCT) applied to image compression, developed in a **CUDA C** environment. The project extends the approximation methodology introduced by Haweel et al. (2016), introducing advanced High Performance Computing (HPC) strategies to maximize computational efficiency on modern hardware architectures.

The work associated with this code was published and presented within the workshops of the **[ICCSA 2026](https://doi.org/10.1007/978-3-032-30530-5_12)** conference.


---

## Implemented Algorithms

The repository includes four variants of the compression algorithm to allow for in-depth benchmarking:

1. **`cublasDCT`**: A native/naïve implementation that computes the DCT on individual $8 \times 8$ sub-blocks using the standard `cuBLAS` library.
2. **`cublasDCTv2`**: An optimized version based on `cuBLAS` that processes the entire image matrix in a single global operation, reducing kernel invocation overhead.
3. **`fastApprDCT`**: Parallel GPU implementation of the approximation algorithm introduced by Haweel et al. in 2016.
4. **`HpApprDCT` (Proposed Algorithm)**: The core version of the project, written entirely from scratch in CUDA C and optimized through advanced memory and thread management techniques.

---

## Advanced Optimization Strategies (`HpApprDCT`)

The proposed **`HpApprDCT`** algorithm achieves state-of-the-art performance thanks to the synergistic combination of several GPU hardware optimization techniques:

* **Optimized Thread-to-Block Mapping**: Unlike previous approaches where a single thread processed an entire row of coefficients across multiple sub-blocks, `HpApprDCT` adopts a 1-to-1 mapping. An $8 \times 8$ thread block is assigned to a single $8 \times 8$ image sub-block. An entire row of threads computes a row of coefficients in parallel, maximizing Streaming Multiprocessor (SM) occupancy.
* **Coalesced Memory Access (Memory Coalescing)**: The transfer of image matrix data from global memory to shared memory occurs in a coalesced manner. Each thread copies a single element contiguously, eliminating bottlenecks related to global memory bandwidth.
* **Use of Constant Memory (Constant Memory)**: All read-only matrices that remain unchanged during execution (the standard quantization matrix $Q$ and the orthogonal and sparse transformation matrix $T$) have been allocated in the GPU's constant memory. This drastically reduces access latency thanks to the automatic broadcasting mechanism to thread warps.
* **Shared Memory Synchronization**: Variables and intermediate results of the two-dimensional DCT computation ($T \cdot X_b \cdot T^T$) are temporarily stored in `shared_memory`. An explicit synchronization barrier (`__syncthreads()`) ensures that intermediate results are consistent for all threads before the second matrix multiplication, minimizing thread divergence.

---

## Experimental Results and Benchmarks

Performance tests were conducted on the following reference hardware:
* **GPU**: NVIDIA Tesla T4 (2560 CUDA Cores, 16 GB GDDR6)
* **CPU**: Intel Xeon @ 2.00GHz
* **Software**: CUDA Toolkit 12.5, `nvcc` compiler with `-O3` flag

### 1. Execution Times (in milliseconds)
Benchmarks were executed by averaging 100 independent computations on images of increasing sizes from $256 \times 256$ up to $8192 \times 8192$:

| Image Size | DCT on CPU (Sequential) | fastApprDCT (GPU) | HpApprDCT (Proposed) |
| :---: | :---: | :---: | :---: |
| **$256 \times 256$** | 4.7 ms | 0.28 ms | **0.07 ms** |
| **$512 \times 512$** | 17.9 ms | 0.33 ms | **0.12 ms** |
| **$1024 \times 1024$** | 72.8 ms | 0.61 ms | **0.30 ms** |
| **$2048 \times 2048$** | 291.7 ms | 1.65 ms | **1.04 ms** |
| **$4096 \times 4096$** | 1255.1 ms | 5.80 ms | **4.00 ms** |
| **$8192 \times 8192$** | 5005.1 ms | 20.00 ms | **14.70 ms** |

### 2. Speedup Gained
* A remarkable acceleration of **340x** compared to the sequential CPU version for high-resolution images ($8192 \times 8192$).
* Reduction of processing times by up to **31%** compared to the `fastApprDCT` CUDA algorithm.
* Reduction of execution time by over **96%** compared to using the standard `cuBLAS` library (`cublasDCTv2`).

### 3. Accuracy Evaluation (*Circuit* Image)
Analysis of the error (PEEN - Percentage Error Energy Norm and MSE - Mean Squared Error) and the compression factor when varying the number of retained coefficients with respect to the standard quantization matrix:

| Metric / Coefficients | 6 | 7 | 8 | 9 | 10 | Standard |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PEEN (%)** | 9.92 | 8.34 | 8.25 | 8.13 | 6.07 | **4.66** |
| **MSE** | 79.99 | 56.67 | 55.32 | 53.80 | 30.04 | **17.67** |
| **Compr. Factor** | 1.49 | 1.49 | 1.40 | 1.35 | 1.35 | **1.29** |

---

## Compilation and Usage

The code is written exclusively in native **standard CUDA C** and is fully *forward-compatible* with subsequent generations of GPU architectures (Ampere, Hopper, Ada Lovelace, Blackwell, etc.).

### Prerequisites
* NVIDIA CUDA Toolkit (v12.0 or later recommended)
* Host-compatible compiler (e.g., `gcc` for Linux)
* `libjpeg` library (required for algorithm using I/O image modules)

### Compilation via Terminal
To compile the project while optimizing the binary for a specific architecture (e.g., Turing `sm_75` or Ampere `sm_80`), run:

```bash
## Proposed Algorithm
nvcc benchmark_newAppr.cu -o benchmark_newAppr.out -arch=sm_75 -O3
nvcc benchmark_newAppr.cu -o benchmark_newAppr.out -arch=sm_80 -O3

## Algorithms that require cublass
nvcc benchmark_cublass_2.cu -o benchmark_cublass_2.out --library cublas -arch=sm_75 -O3
nvcc benchmark_cublass_2.cu -o benchmark_cublass_2.out --library cublas -arch=sm_75 -O3
```
