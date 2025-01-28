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
bool arrays_are_close(const float* arr1, const float* arr2, const size_t size, const float tol)
{
    for (size_t i = 0; i < size; i++) {
        if (fabs(arr1[i] - arr2[i]) > tol) {
            return false;
        }
    }
    return true;
}

// Load the "*filename" image and return its pointer. (fills width, height and channels vars)
static unsigned char *load_jpeg_as_matrix(const char *filename, int *width, int *height, int *channels)
{
    // Jpeg decompression struct
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    // Open the file with read option
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

// Save the *image_matrix with name *filename
int save_grayscale_jpeg(const char *filename, unsigned char *image_matrix, const int width, const int height, const int quality)
{
    // Jpeg compression struct
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    // Set up error handling
    cinfo.err = jpeg_std_error(&jerr);

    // Jpeg object for compression
    jpeg_create_compress(&cinfo);

    // Open the file with write option
    FILE *outfile = fopen(filename, "wb");
    if (!outfile)
    {
        fprintf(stderr, "Error: Unable to open file %s for writing\n", filename);
        return 0;
    }

    // Specify the destination of the data (the output file)
    jpeg_stdio_dest(&cinfo, outfile);

    // Write the JPEG header for image info
    cinfo.image_width = width;            // Larghezza in pixel
    cinfo.image_height = height;          // Altezza in pixel
    cinfo.input_components = 1;           // Numero di canali (1 per scala di grigi)
    cinfo.in_color_space = JCS_GRAYSCALE; // Colore: scala di grigi

    // Set defaults parameters and quality
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    // Start compression
    jpeg_start_compress(&cinfo, TRUE);

    // Write scanlines into the matrix row by row
    while (cinfo.next_scanline < cinfo.image_height)
    {
        unsigned char *row_pointer = &image_matrix[cinfo.next_scanline * width];
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    // Finish comrpession and clean up res
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);

    return 1;
}