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
/**
 * Validates whether the results of stage 1 have been calculated correctly
 * Success or failure will be printed to the console
 *
 * @param input_image Host pointer to (a copy of) the input image provided to stage 1
 * @param test_histograms Host pointer to a 2-dimensional array of histograms to be checked
 * @param max_contrast The most common contrast value as calculated and returned by stage 1
 *
 * @note If test_histograms does not match the same memory layout as cpu.c, this may cause an access violation
 */
void validate_histogram(const Image *input_image, Histogram_uint **test_histograms, int max_contrast);
/**
 * Calculate the results of stage 1 from the input_image
 *
 * @param input_image Host pointer to (a copy of) the input image provided to stage 1
 * @param histograms Host pointer to a pre-allocated 2-dimensional array of histograms
 * @return The most common contrast value found in the image
 *
 * @note If histograms does not match the same memory layout as cpu.c, this may cause an access violation
 */
int skip_histogram(const Image *input_image, Histogram_uint** histograms);

///
/// Stage 2 helpers
///
/**
 * Validates whether each histograms[][]->limited_histogram of stage 2 has been calculated correctly
 * Success or failure will be printed to the console
 *
 * @param TILES_X The number of histograms in the first dimension of test_histograms
 *        Also known as, the horizontal number of tiles in the image (input_image->width / TILE_SIZE)
 * @param TILES_Y The number of histograms in the second dimension of test_histograms
 *        Also known as, the vertical number of tiles in the image (input_image->height / TILE_SIZE)
 * @param histograms Host pointer to a 2-dimensional array of histograms from stage 1
 * @param test_limited_histograms Host pointer to a 2-dimensional array of limited histograms to be checked
 *
 * @note If test_histograms does not match the same memory layout as cpu.c, this may cause an access violation
 */
void validate_limited_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram_uint** histograms, Histogram_uint** test_limited_histograms);
/**
 * Calculate histograms[][]->limited_histogram of stage 2 using histograms[][]->histogram
 * The result is applied to the parameter histograms
 *
 * @param TILES_X The number of histograms in the first dimension of test_histograms
 *        Also known as, the horizontal number of tiles in the image (input_image->width / TILE_SIZE)
 * @param TILES_Y The number of histograms in the second dimension of test_histograms
 *        Also known as, the vertical number of tiles in the image (input_image->height / TILE_SIZE)
 * @param histograms Host pointer to a 2-dimensional array of histograms from stage 1
 * @param limited_histograms Host pointer to a pre-allocated 2-dimensional array of histograms
 *
 * @note If histograms/limited_histograms do not match the same memory layout as cpu.c, this may cause an access violation
 * @note Using this method will not calculate the per-histogram lost_contrast value
 */
void skip_limited_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram_uint** histograms, Histogram_uint** limited_histograms);
/**
 * Validates whether each histograms[][]->cumulative_histogram of stage 2 has been calculated correctly
 * Success or failure will be printed to the console
 *
 * @param TILES_X The number of histograms in the first dimension of limited_histograms
 *        Also known as, the horizontal number of tiles in the image (input_image->width / TILE_SIZE)
 * @param TILES_Y The number of histograms in the second dimension of limited_histograms
 *        Also known as, the vertical number of tiles in the image (input_image->height / TILE_SIZE)
 * @param limited_histograms Host pointer to a 2-dimensional array of limited histograms
 * @param test_cumulative_histograms Host pointer to a pre-allocated 2-dimensional array of cumulative histograms to be checked
 *
 * @note If limited_histograms/test_cumulative_histograms do not match the same memory layout as cpu.c, this may cause an access violation
 */
void validate_cumulative_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram_uint** limited_histograms, Histogram_uint** test_cumulative_histograms);
/**
 * Calculate histograms[][]->cumulative_histogram of stage 2 using histograms[][]->limited_histogram
 * The result is applied to the parameter histograms
 *
 * @param TILES_X The number of histograms in the first dimension of test_histograms
 *        Also known as, the horizontal number of tiles in the image (input_image->width / TILE_SIZE)
 * @param TILES_Y The number of histograms in the second dimension of test_histograms
 *        Also known as, the vertical number of tiles in the image (input_image->height / TILE_SIZE)
 * @param limited_histograms Host pointer to a 2-dimensional array of limited histograms
 * @param cumulative_histograms Host pointer to a pre-allocated 2-dimensional array of cumulative histograms
 *
 * @note If limited_histograms/cumulative_histograms do not match the same memory layout as cpu.c, this may cause an access violation
 * @note Using this method will not calculate the per-histogram cdf_min value
 */
void skip_cumulative_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram_uint** limited_histograms, Histogram_uint** cumulative_histograms);
/**
 * Validates whether eequalsied_histograms of stage 2 have been calculated correctly
 * from the histograms from stage 1
 * Success or failure will be printed to the console
 *
 * @param TILES_X The number of histograms in the first dimension of test_histograms
 *        Also known as, the horizontal number of tiles in the image (input_image->width / TILE_SIZE)
 * @param TILES_Y The number of histograms in the second dimension of test_histograms
 *        Also known as, the vertical number of tiles in the image (input_image->height / TILE_SIZE)
 * @param histograms Host pointer to a 2-dimensional array of histograms from stage 1
 * @param test_equalisied_histograms Host pointer to a 2-dimensional array of equalised histograms to be checked
 *
 * @note If histograms/test_equalisied_histograms does not match the same memory layout as cpu.c, this may cause an access violation
 */
void validate_equalised_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram_uint** histograms, Histogram_uchar **test_equalisied_histograms);
/**
 * Calculate equalsied_histograms of stage 2 using histograms from stage 1 (This skips all of stage 2)
 * The result is applied to the parameter histograms
 *
 * @param TILES_X The number of histograms in the first dimension of test_histograms
 *        Also known as, the horizontal number of tiles in the image (input_image->width / TILE_SIZE)
 * @param TILES_Y The number of histograms in the second dimension of test_histograms
 *        Also known as, the vertical number of tiles in the image (input_image->height / TILE_SIZE)
 * @param histograms Host pointer to a 2-dimensional array of histograms from stage 1
 * @param equalisied_histograms Host pointer to a pre-allocated  2-dimensional array of equalised histograms
 *
 * @note If histograms/equalisied_histograms do not match the same memory layout as cpu.c, this may cause an access violation
 * @note This method calculates the per-histogram lost_contrast and cdf_min values itself
 */
void skip_equalised_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram_uint** histograms, Histogram_uchar** equalisied_histograms);

///
/// Stage 3 helpers
///
/**
 * Validates whether the output image of stage 3 has been calculated correctly
 * Success or failure will be printed to the console
 *
 * @param input_image Host pointer to (a copy of) the input image provided to stage 1
 * @param equalised_histograms Host pointer to a 2-dimensional array of equalised histograms
 * @param test_output_image Host pointer to a pre-allocated image for output
 *
 * @note If any of the input parameters do not point to memory matching the layout of cpu.c, this may cause an access violation
 */
void validate_interpolate(const Image *input_image, Histogram_uchar** equalised_histograms, Image *test_output_image);
/**
 * Calculate the output image of stage 3 using histograms[][]->equalised_histogram from stage 2 and the input image
 * The result is applied to the parameter histograms
 *
 * @param input_image Host pointer to (a copy of) the input image provided to stage 1
 * @param equalised_histograms Host pointer to a 2-dimensional array of equalised histograms
 * @param output_image Host pointer to a pre-allocated image for output
 *
 * @note If any of the input parameters do not point to memory matching the layout of cpu.c, this may cause an access violation
 */
void skip_interpolate(const Image *input_image, Histogram_uchar** equalised_histograms, Image *output_image);

///
/// These are used for reporting whether timing is invalid due to helper use
///
int getSkipUsed();
int getStage1SkipUsed();
int getStage2SkipUsed();
int getStage3SkipUsed();

#ifdef __cplusplus
}
#endif

#if defined(_DEBUG) || defined(DEBUG)
#define VALIDATION
#endif

#endif  // __helper_h__
