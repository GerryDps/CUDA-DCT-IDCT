#ifndef PROJECT_CUDA_UTILS_H
#define PROJECT_CUDA_UTILS_H

// Convert an unsigned char image matrix to an float image matrix
void convertToFloat(const unsigned char *input, float *output, const size_t size);
// Convert a float image matrix to unsigned char image matrix
void convertToUnsignedChar(const float *image_float, unsigned char *image_char, const size_t size);
// Check if arr1=+-=arr2 +-tolerance
bool arrays_are_close(const float* arr1, const float* arr2, const size_t size, const float tol);

// Load the "*filename" image and return its pointer. (fills width, height and channels vars)
unsigned char *load_jpeg_as_matrix(const char *filename, int *width, int *height, int *channels);
// Save the *image_matrix with name *filename
int save_grayscale_jpeg(const char *filename, unsigned char *image_matrix, const int width, const int height, const int quality);

#endif //PROJECT_CUDA_UTILS_H
