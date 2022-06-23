#include "cuda.cuh"
#include "helper.h"

#include <cstring>

///
/// Algorithm storage
///
// Pointer to array of histograms on the device
Histogram_uint* d_histograms;
Histogram_uint* d_limited_histograms;
Histogram_uint* d_cumulative_histograms;
Histogram_uchar* d_equalised_histograms;
// Host copy of input image
Image cuda_input_image;
// Pointer to device image data buffer, for storing the input image, this must be passed to a kernel to be used on device
unsigned char* d_input_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;
// Host copy of image tiles in each dimension
unsigned int cuda_TILES_X, cuda_TILES_Y;

void cuda_begin(const Image *input_image) {
    // These are suggested CUDA memory allocations that match the CPU implementation
    // If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference

    cuda_TILES_X = input_image->width / TILE_SIZE;
    cuda_TILES_Y = input_image->height / TILE_SIZE;

    // Allocate histogram per tile
    CUDA_CALL(cudaMalloc(&d_histograms, cuda_TILES_X * cuda_TILES_Y * sizeof(Histogram_uint)));
    CUDA_CALL(cudaMalloc(&d_limited_histograms, cuda_TILES_X * cuda_TILES_Y * sizeof(Histogram_uint)));
    CUDA_CALL(cudaMalloc(&d_cumulative_histograms, cuda_TILES_X * cuda_TILES_Y * sizeof(Histogram_uint)));
    CUDA_CALL(cudaMalloc(&d_equalised_histograms, cuda_TILES_X * cuda_TILES_Y * sizeof(Histogram_uchar)));

    // Zero all the histograms
    CUDA_CALL(cudaMemset(d_histograms, 0, cuda_TILES_X * cuda_TILES_Y * sizeof(Histogram_uint)));
    CUDA_CALL(cudaMemset(d_limited_histograms, 0, cuda_TILES_X * cuda_TILES_Y * sizeof(Histogram_uint)));
    CUDA_CALL(cudaMemset(d_cumulative_histograms, 0, cuda_TILES_X * cuda_TILES_Y * sizeof(Histogram_uint)));
    CUDA_CALL(cudaMemset(d_equalised_histograms, 0, cuda_TILES_X * cuda_TILES_Y * sizeof(Histogram_uchar)));

    const size_t image_data_size = input_image->width * input_image->height * sizeof(unsigned char);
    // Allocate copy of input image
    cuda_input_image = *input_image;
    cuda_input_image.data = (unsigned char*)malloc(image_data_size);
    memcpy(cuda_input_image.data, input_image->data, image_data_size);

    // Allocate and fill device buffer for storing input image data
    CUDA_CALL(cudaMalloc(&d_input_image_data, image_data_size));
    CUDA_CALL(cudaMemcpy(d_input_image_data, input_image->data, image_data_size, cudaMemcpyHostToDevice));

    // Allocate device buffer for storing output image data
    CUDA_CALL(cudaMalloc(&d_output_image_data, image_data_size));
}
int cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // int max_contrast = skip_histogram(&input_image, histograms)

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // validate_histogram(&input_image, histograms, max_contrast);
#endif
    return -1;
}
void cuda_stage2() {
    // Optionally during development call the skip function/s with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // skip_limited_histogram(TILES_X, TILES_Y, histograms, limited_histograms)
    // skip_cumulative_histogram(TILES_X, TILES_Y, limited_histograms, cumulative_histograms)
    // skip_equalised_histogram(TILES_X, TILES_Y, histograms, equalisied_histograms)

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // Note: Only validate_equalised_histogram() MUST be uncommented, the others are optional
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // validate_limited_histogram(TILES_X, TILES_Y, histograms, test_limited_histograms);
    // validate_cumulative_histogram(TILES_X, TILES_Y, limited_histograms, test_cumulative_histograms);
    // validate_equalised_histogram(TILES_X, TILES_Y, histograms, test_equalisied_histograms);
#endif    
}
void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // skip_interpolate(&input_image, equalised_histograms, &output_image)

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // validate_interpolate(&input_image, equalised_histograms, &output_image);
#endif    
}
void cuda_end(Image *output_image) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    output_image->width = cuda_input_image.width;
    output_image->height = cuda_input_image.height;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, output_image->width * output_image->height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    free(cuda_input_image.data);
    CUDA_CALL(cudaFree(d_output_image_data));
    CUDA_CALL(cudaFree(d_input_image_data));
    CUDA_CALL(cudaFree(d_equalised_histograms));
    CUDA_CALL(cudaFree(d_cumulative_histograms));
    CUDA_CALL(cudaFree(d_limited_histograms));
    CUDA_CALL(cudaFree(d_histograms));
}