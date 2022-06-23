#include "cuda.cuh"

#include <cstdlib>
#include <cstring>

void cuda_begin(const Image *input_image) {
    //TODO Allocate memory and copy data to device
}

int cuda_CLAHE() {
    //TODO
    return -1;
}

void cuda_end(Image *output_image) {
    // TODO copy the result image from device and free cuda memory
}