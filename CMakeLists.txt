cmake_minimum_required(VERSION 3.27)
project(project_CUDA CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(project_CUDA main.cu
        cudaonly.cu
        test.cu
)

set_target_properties(project_CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
