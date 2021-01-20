#ifndef __cuda_cuh__
#define __cuda_cuh__

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

/**
 * The initialisation function for the CUDA CLAHE implemention
 * Memory allocation and initialisation occurs here, so that it can be timed separate to the algorithm
 * @param input_image Pointer to a constant struct containing the image to be processed
 */
void cuda_begin(const Image *input_image);
/**
 * Create a locatlised histogram for each tile of the image
 */
int cuda_stage1();
/**
 * Equalise the histograms
 */
void cuda_stage2();
/**
 * Interpolate the histograms to construct the contrast enhanced image for output
 */
void cuda_stage3();
/**
 * The cleanup and return function for the CPU CLAHE implemention
 * Memory should be freed, and the final image copied to output_image
 * @param output_image Pointer to a struct to store the final image to be output
 */
void cuda_end(Image *output_image);


/**
 * Error check function for safe CUDA API calling
 * Wrap all calls to CUDA API functions with CUDA_CALL() to catch errors on failure
 * e.g. CUDA_CALL(cudaFree(myPtr));
 * CUDA_CHECk() can also be used to perform error checking after kernel launches
 * e.g. CUDA_CHECK("kernel launch 1")
 */
#ifdef _DEBUG
#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_CHECK(location) { gpuAssert(cudaDeviceSynchronize(), location, -1); }
#else
#define CUDA_CALL(ans) { ans; }
#define CUDA_CHECK(location) { gpuAssert(cudaPeekAtLastError(), location, -1); }
#endif
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        if (line >= 0) {
            fprintf(stderr, "CUDA Error: %s(%d): %s", file, line, cudaGetErrorString(code));
        } else {
            fprintf(stderr, "CUDA Error occurred during %s: %s", file, cudaGetErrorString(code));
        }
        exit(EXIT_FAILURE);
    }
}

#endif // __cuda_cuh__
