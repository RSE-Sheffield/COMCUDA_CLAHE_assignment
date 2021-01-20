#ifndef __openmp_h__
#define __openmp_h__

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The initialisation function for the OpenMP CLAHE implemention
 * Memory allocation and initialisation occurs here, so that it can be timed separate to the algorithm
 * @param input_image Pointer to a constant struct containing the image to be processed
 */
void openmp_begin(const Image *input_image);
/**
 * Create a locatlised histogram for each tile of the image
 */
int openmp_stage1();
/**
 * Equalise the histograms
 */
void openmp_stage2();
/**
 * Interpolate the histograms to construct the contrast enhanced image for output
 */
void openmp_stage3();
/**
 * The cleanup and return function for the CPU CLAHE implemention
 * Memory should be freed, and the final image copied to output_image
 * @param output_image Pointer to a struct to store the final image to be output
 */
void openmp_end(Image *output_image);

#ifdef __cplusplus
}
#endif

#endif // __openmp_h__
