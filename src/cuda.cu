#include "cuda.cuh"

void cuda_begin(const Image *input_image) {
    
}
int cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // int max_contrast = skip_histogram(input_image, histograms)

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // validate_histogram(input_image, histograms, max_contrast);
#endif
    return -1;
}
void cuda_stage2() {
    // Optionally during development call the skip function/s with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // skip_limited_histogram(TILES_X, TILES_Y, histograms)
    // skip_cumulative_histogram(TILES_X, TILES_Y, histograms)
    // skip_equalised_histogram(TILES_X, TILES_Y, histograms)

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // validate_limited_histogram(TILES_X, TILES_Y, test_histograms);
    // validate_cumulative_histogram(TILES_X, TILES_Y, test_histograms);
    // validate_equalised_histogram(TILES_X, TILES_Y, test_histograms);
#endif    
}
void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // skip_interpolate(input_image, histograms, output_image)

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // validate_interpolate(input_image, histograms, output_image);
#endif    
}
void cuda_end(Image *output_image) {
    
}