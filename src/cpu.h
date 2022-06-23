#ifndef __cpu_h__
#define __cpu_h__

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The initialisation function for the CPU CLAHE implementation
 * Memory allocation and initialisation occurs here, so that it can be timed separate to the algorithm
 * @param input_image Pointer to a constant struct containing the image to be processed
 */
void cpu_begin(const Image *input_image);
/**
 * CLAHE reference algorithm
 * Create a localised histogram for each tile of the image
 * Equalise the histograms
 * Interpolate the histograms to construct the contrast enhanced image for output
 * @return The most common contrast value
 */
int cpu_CLAHE();
/**
 * The cleanup and return function for the CPU CLAHE implementation
 * Memory should be freed, and the final image copied to output_image
 * @param output_image Pointer to a struct to store the final image to be output, output_image->data is pre-allocated
 */
void cpu_end(Image *output_image);

#ifdef __cplusplus
}
#endif

#endif // __cpu_h__
