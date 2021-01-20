#ifndef __common_h__
#define __common_h__

#include "config.h"

/**
 * This structure represents a single channel image (e.g. greyscale)
 * It contains the data required by the stb image read/write functions
 */
struct Image {
   /**
    * Array of pixel data of the image, 1 unsigned char per pixel channel
    * Pixels ordered left to right, top to bottom
    * There is no stride, this is a compact storage
    */
    unsigned char *data;
    /**
     * Image width and height
     */
    int width, height;
};
typedef struct Image Image;
/**
 * This structure represents a collection of histograms used throughout the algorithm
 */
struct Histogram {
    /**
     *
     */
    unsigned int histogram[PIXEL_RANGE];
    /**
     *
     */
    unsigned int limited_histogram[PIXEL_RANGE];
    /**
     *
     */
    unsigned int cumulative_histogram[PIXEL_RANGE];
    /**
     *
     */
    unsigned char equalised_histogram[PIXEL_RANGE];
};
typedef struct Histogram Histogram;

#endif  // __common_h__
