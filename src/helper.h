#ifndef __helper_h__
#define __helper_h__

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * These helper methods can be use to check validity of individual stages of your algorithm
 * Pass the required component to the validate_?() methods for it's validation status to be printed to console
 * Pass the required components to the skip_?() methods to have the reference implementation perform the step.
 * For the CUDA algorithm, this will require copying data back to the host. You CANNOT pass device pointers to these methods.
 *
 * Pointers passed to helper methods must point to memory which has been allocated in the same format as CPU.c
 *
 * Some images will have limited/no errors when performed incorrectly, it's best to validate with a wide range of images.
 *
 * Do not use these methods during benchmark runs, as they will invalidate the timing.
 */

///
/// Stage 1 helpers
///

void validate_histogram(const Image *input_image, Histogram **test_histograms, int max_contrast);
int skip_histogram(const Image *input_image, Histogram **histograms);

///
/// Stage 2 helpers
///

void validate_limited_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram **test_histograms);
void skip_limited_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram **histograms);
void validate_cumulative_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram **test_histograms);
void skip_cumulative_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram **histograms);
void validate_equalised_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram **test_histograms);
void skip_equalised_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram **histograms);

///
/// Stage 3 helpers
///

void validate_interpolate(const Image *input_image, Histogram **histograms, Image *test_output_image);
void skip_interpolate(const Image *input_image, Histogram **histograms, Image *output_image);

///
/// These are used for reporting whether timing is invalid due to helper use
///
///
int getSkipUsed();
int getStage1SkipUsed();
int getStage2SkipUsed();
int getStage3SkipUsed();

#ifdef __cplusplus
}
#endif

#endif  // __helper_h__
