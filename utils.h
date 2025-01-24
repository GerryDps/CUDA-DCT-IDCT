#ifndef PROJECT_CUDA_UTILS_H
#define PROJECT_CUDA_UTILS_H

void convertToFloat(unsigned char *input, float *output, int size);
void convertToUnsignedChar(const float *image_float, unsigned char *image_char, int size);


#endif //PROJECT_CUDA_UTILS_H
