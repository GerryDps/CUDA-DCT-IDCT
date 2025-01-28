#include <math.h>
#include "utils.cuh"
// Convert an unsigned char image matrix to an float image matrix
void convertToFloat(const unsigned char *input, float *output, const size_t size)
{
    for (int i = 0; i < size; i++) {
        output[i] = (float)input[i];
    }
}

// Convert a float image matrix to unsigned char image matrix
void convertToUnsignedChar(const float *image_float, unsigned char *image_char, const size_t size)
{
    for (int i = 0; i < size; i++) {
        image_char[i] = (unsigned char)fminf(fmaxf(image_float[i], 0.0f), 255.0f); // Clamp tra 0 e 255
        //image_char[i] = (unsigned char)image_float[i]; // Clamp tra 0 e 255
    }
}

// Check if arr1=+-=arr2 +-tolerance
bool arrays_are_close(const float* arr1, const float* arr2, const size_t size, const float tol){
    for (size_t i = 0; i < size; i++) {
        if (fabs(arr1[i] - arr2[i]) > tol) {
            return false;
        }
    }
    return true;
}