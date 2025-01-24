#ifndef PROJECT_CUDA_UTILS_H
#define PROJECT_CUDA_UTILS_H

void convertToFloat(const unsigned char *input, float *output, const size_t size);
void convertToUnsignedChar(const float *image_float, unsigned char *image_char, const size_t size);
bool arrays_are_close(const float* arr1, const float* arr2, const size_t size, const float tol);



#endif //PROJECT_CUDA_UTILS_H
