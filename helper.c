#include "helper.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"

///
/// Utility Methods
///

/**
 * Returns x * a + y * (1.0 - a)
 * e.g. The linear blend of x and y using the floating-point value a.
 * The value for a is not restricted to the range [0, 1].
 */
inline unsigned char mix_uc(unsigned char x, unsigned char y, float a) {
    return (unsigned char)(x * a + y * (1.0f - a));
}
inline float mix_f(float x, float y, float a) {
    return x * a + y * (1.0f - a);
}
/**
 * Returns the offset of the tile to be used for interpolation, based on the position of a pixel within the tile
 * This is specific to each axis
 * @param i Position of the pixel within the tile
 */
inline int lerp_direction(unsigned int i) {
    return i < HALF_TILE_SIZE ? -1 : 1;
}
/**
 * Returns the interpolation weight, based on the position of a pixel within the tile
 * This is specific to each axis
 * @param i Position of the pixel within the tile
 */
inline float lerp_weight(unsigned int i) {
    return  (i < HALF_TILE_SIZE ? HALF_TILE_SIZEf + i : 1.5f * TILE_SIZE - i) / TILE_SIZEf;
}

int skip_histogram_used = -1;
void validate_histogram(const Image *input_image, Histogram **test_histograms) {
    const unsigned int TILES_X = input_image->width / TILE_SIZE;
    const unsigned int TILES_Y = input_image->height / TILE_SIZE;
    // Allocate and generate our own internal histogram
    Histogram **histograms = (Histogram **)malloc(TILES_X * sizeof(Histogram*));
    histograms[0] = (Histogram *)malloc(TILES_X * TILES_Y * sizeof(Histogram));
    memset(histograms[0],0, TILES_X * TILES_Y * sizeof(Histogram));
    for (unsigned int t_x = 1; t_x < TILES_X; ++t_x) {
        histograms[t_x] = histograms[0] + t_x * TILES_Y;         
    }
    skip_histogram(input_image, histograms);
    // Validate and report result
    unsigned int bad_tiles = 0;
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            for (unsigned int i = 0; i < PIXEL_RANGE; ++ i) {
                if (test_histograms[t_x][t_y].histogram[i] != histograms[t_x][t_y].histogram[i]) {
                    bad_tiles++;
                    break;
                }
            }
        }
    }
    if (bad_tiles) {
        fprintf(stderr, "validate_histogram() found %d/%u tiles contain invalid histograms.\n", bad_tiles, TILES_X * TILES_Y);
    } else {
        fprintf(stderr, "validate_histogram() found no errors! (%u tiles were correct)\n", TILES_X * TILES_Y);
    }

    // Release internal histogram
    free(histograms[0]);
    free(histograms);
}
void skip_histogram(const Image *input_image, Histogram **histograms) {
    const unsigned int TILES_X = input_image->width / TILE_SIZE;
    const unsigned int TILES_Y = input_image->height / TILE_SIZE;
    // Generate histogram per tile
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            const unsigned int tile_offset = (t_y * TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE); 
            // For each pixel within the tile
            for (int p_x = 0; p_x < TILE_SIZE; ++p_x) {
                for (int p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    // Load pixel
                    const unsigned int pixel_offset = (p_y * input_image->width + p_x); 
                    const unsigned char pixel = input_image->data[tile_offset + pixel_offset];
                    histograms[t_x][t_y].histogram[pixel]++;
                }
            }
        }
    }
    skip_histogram_used++;
}

int skip_limited_histogram_used = -1;
void validate_limited_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram **test_histograms) {
    // Allocate, copy and generate our own internal histogram
    Histogram **histograms = (Histogram **)malloc(TILES_X * sizeof(Histogram*));
    histograms[0] = (Histogram *)malloc(TILES_X * TILES_Y * sizeof(Histogram));
    memset(histograms[0],0, TILES_X * TILES_Y * sizeof(Histogram));
    for (unsigned int t_x = 1; t_x < TILES_X; ++t_x) {
        histograms[t_x] = histograms[0] + t_x * TILES_Y;         
    }    
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            memcpy(histograms[t_x][t_y].histogram, test_histograms[t_x][t_y].histogram, PIXEL_RANGE * sizeof(unsigned int));
        }
    }
    skip_limited_histogram(TILES_X, TILES_Y, histograms);
    // Validate and report result
    unsigned int bad_histograms = 0;
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            for (unsigned int i = 0; i < PIXEL_RANGE; ++ i) {
                if (test_histograms[t_x][t_y].limited_histogram[i] != histograms[t_x][t_y].limited_histogram[i]) {
                    bad_histograms++;
                    break;
                }
            }
        }
    }
    if (bad_histograms) {
        fprintf(stderr, "validate_limited_histogram() found %d/%u incorrect limited histograms.\n", bad_histograms, TILES_X * TILES_Y);
    } else {
        fprintf(stderr, "validate_limited_histogram() found no errors! (%u limited histograms were correct)\n", TILES_X * TILES_Y);
    }
    // Release internal histogram
    free(histograms[0]);
    free(histograms);
}
void skip_limited_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram **histograms) {
    // For each histogram
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            // Clamp where count exceeds ABSOLUTE_CONTRAST_LIMIT
            int extra_contrast = 0;
            for (unsigned int i = 0; i < PIXEL_RANGE; ++i) {
                if (histograms[t_x][t_y].histogram[i] > ABSOLUTE_CONTRAST_LIMIT) {
                    extra_contrast += histograms[t_x][t_y].histogram[i] - ABSOLUTE_CONTRAST_LIMIT;
                    histograms[t_x][t_y].limited_histogram[i] = ABSOLUTE_CONTRAST_LIMIT;
                } else {
                    histograms[t_x][t_y].limited_histogram[i] = histograms[t_x][t_y].histogram[i];
                }
            }
            // int lost_contrast = 0;
            if (extra_contrast > PIXEL_RANGE) {
                const int bonus_contrast = extra_contrast / PIXEL_RANGE;  // integer division is fine here
                // lost_contrast = extra_contrast % PIXEL_RANGE;
                for (int i = 0; i < PIXEL_RANGE; ++i) {
                    histograms[t_x][t_y].limited_histogram[i] += bonus_contrast;
                }
            }
        }
    }
    skip_limited_histogram_used++;
}
int skip_cumulative_histogram_used = -1;
void validate_cumulative_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram **test_histograms) {
    // Allocate, copy and generate our own internal histogram
    Histogram **histograms = (Histogram **)malloc(TILES_X * sizeof(Histogram*));
    histograms[0] = (Histogram *)malloc(TILES_X * TILES_Y * sizeof(Histogram));
    memset(histograms[0],0, TILES_X * TILES_Y * sizeof(Histogram));
    for (unsigned int t_x = 1; t_x < TILES_X; ++t_x) {
        histograms[t_x] = histograms[0] + t_x * TILES_Y;         
    }    
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            // memcpy(histograms[t_x][t_y].histogram, test_histograms[t_x][t_y].histogram, PIXEL_RANGE * sizeof(unsigned int));
            memcpy(histograms[t_x][t_y].limited_histogram, test_histograms[t_x][t_y].limited_histogram, PIXEL_RANGE * sizeof(unsigned int));
        }
    }
    skip_cumulative_histogram(TILES_X, TILES_Y, histograms);
    // Validate and report result
    unsigned int bad_histograms = 0;
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            for (unsigned int i = 0; i < PIXEL_RANGE; ++ i) {
                if (test_histograms[t_x][t_y].cumulative_histogram[i] != histograms[t_x][t_y].cumulative_histogram[i]) {
                    bad_histograms++;
                    break;
                }
            }
        }
    }
    if (bad_histograms) {
        fprintf(stderr, "validate_cumulative_histogram() found %d/%u incorrect cumulative histograms.\n", bad_histograms, TILES_X * TILES_Y);
    } else {
        fprintf(stderr, "validate_cumulative_histogram() found no errors! (%u cumulative histograms were correct)\n", TILES_X * TILES_Y);
    }
    // Release internal histogram
    free(histograms[0]);
    free(histograms);
}
void skip_cumulative_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram **histograms) {
    // For each histogram
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            // Find cdf_min and convert histogram to cumulative
            // This is essentially a scan
            // unsigned int cdf_min = 0;
            histograms[t_x][t_y].cumulative_histogram[0] = histograms[t_x][t_y].limited_histogram[0];
            for (unsigned int i = 1; i < PIXEL_RANGE; ++i) {
                histograms[t_x][t_y].cumulative_histogram[i] = histograms[t_x][t_y].cumulative_histogram[i-1] + histograms[t_x][t_y].limited_histogram[i];
                // if (histograms[t_x][t_y].cumulative_histogram[i-1] == 0 && histograms[t_x][t_y].cumulative_histogram[i] != 0) { // Second half of condition is redundant in serial
                //     cdf_min = histograms[t_x][t_y].cumulative_histogram[i];
                // }
            }
        }
    }
    skip_cumulative_histogram_used++;
}
int skip_equalised_histogram_used = -1;
void validate_equalised_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram **test_histograms) {
    // Allocate, copy and generate our own internal histogram
    Histogram **histograms = (Histogram **)malloc(TILES_X * sizeof(Histogram*));
    histograms[0] = (Histogram *)malloc(TILES_X * TILES_Y * sizeof(Histogram));
    memset(histograms[0],0, TILES_X * TILES_Y * sizeof(Histogram));
    for (unsigned int t_x = 1; t_x < TILES_X; ++t_x) {
        histograms[t_x] = histograms[0] + t_x * TILES_Y;         
    }    
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            memcpy(histograms[t_x][t_y].histogram, test_histograms[t_x][t_y].histogram, PIXEL_RANGE * sizeof(unsigned int));
            memcpy(histograms[t_x][t_y].limited_histogram, test_histograms[t_x][t_y].limited_histogram, PIXEL_RANGE * sizeof(unsigned int));
            memcpy(histograms[t_x][t_y].cumulative_histogram, test_histograms[t_x][t_y].cumulative_histogram, PIXEL_RANGE * sizeof(unsigned int));
        }
    }
    skip_equalised_histogram(TILES_X, TILES_Y, histograms);
    // Validate and report result
    unsigned int bad_histograms = 0;
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            for (unsigned int i = 0; i < PIXEL_RANGE; ++ i) {
                if (test_histograms[t_x][t_y].equalised_histogram[i] != histograms[t_x][t_y].equalised_histogram[i]) {
                    bad_histograms++;
                    break;
                }
            }
        }
    }
    if (bad_histograms) {
        fprintf(stderr, "validate_equalised_histogram() found %d/%u incorrect equalised histograms.\n", bad_histograms, TILES_X * TILES_Y);
    } else {
        fprintf(stderr, "validate_equalised_histogram() found no errors! (%u equalised histograms were correct)\n", TILES_X * TILES_Y);
    }
    // Release internal histogram
    free(histograms[0]);
    free(histograms);
}
void skip_equalised_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram **histograms) {
    // For each histogram
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            // Find lost_contrast (This requires the original histogram!)
            int extra_contrast = 0;
            for (unsigned int i = 0; i < PIXEL_RANGE; ++i) {
                if (histograms[t_x][t_y].histogram[i] > ABSOLUTE_CONTRAST_LIMIT) {
                    extra_contrast += histograms[t_x][t_y].histogram[i] - ABSOLUTE_CONTRAST_LIMIT;
                }
            }
            const int lost_contrast = (extra_contrast > PIXEL_RANGE) ? (extra_contrast % PIXEL_RANGE) : 0;
            // Find cdf_min
            unsigned int cdf_min = 0;
            for (unsigned int i = 1; i < PIXEL_RANGE; ++i) {
                if (histograms[t_x][t_y].cumulative_histogram[i-1] == 0 && histograms[t_x][t_y].cumulative_histogram[i] != 0) { // Second half of condition is redundant in serial
                    cdf_min = histograms[t_x][t_y].cumulative_histogram[i];
                    break;
                }
            }
            // Calculate equalised histogram value
            for (unsigned int i = 0; i < PIXEL_RANGE; ++i) {
                float t = roundf(((histograms[t_x][t_y].cumulative_histogram[i] - cdf_min) / (float)(TILE_PIXELS - lost_contrast)) * (float)PIXEL_MAX);
                t = t > PIXEL_MAX ? PIXEL_MAX : t; // indices before cdf_min overflow
                // Clamp value to bounds
                histograms[t_x][t_y].equalised_histogram[i] = (unsigned char)t;
                
            }
        }
    }
    skip_equalised_histogram_used++;
}

int skip_interpolate_used = -1;
void validate_interpolate(const Image *input_image, Histogram **histograms, Image *test_output_image) {
    // Allocate, copy and generate our own internal output image
    Image output_image;
    memcpy(&output_image, input_image, sizeof(Image));
    output_image.data =
        malloc(output_image.width * output_image.height * sizeof(unsigned char));
    
    skip_interpolate(input_image, histograms, &output_image);
    skip_interpolate_used--;
    // Validate and report result
    unsigned int bad_pixels = 0;
    unsigned int close_pixels = 0;
    for (int i = 0; i < output_image.width * output_image.height; ++i) {
        if (output_image.data[i] != test_output_image->data[i]) {
            // Give a +-1 threshold for error (incase fast-math triggers a small difference in places)
            if (output_image.data[i]+1 == test_output_image->data[i] || output_image.data[i]-1 == test_output_image->data[i]) {
                close_pixels++;
            } else {
                bad_pixels++;
            }
        }
    }
    if (bad_pixels) {
        fprintf(stderr, "validate_interpolate() found %d/%u incorrect pixels.\n", bad_pixels, output_image.width * output_image.height);
    } else {
        fprintf(stderr, "validate_interpolate() found no errors! (%u pixels were correct)\n", output_image.width * output_image.height);
    }
    // Release internal output image
    free(output_image.data);    
}
void skip_interpolate(const Image *input_image, Histogram **histograms, Image *output_image) {
    const unsigned int TILES_X = input_image->width / TILE_SIZE;
    const unsigned int TILES_Y = input_image->height / TILE_SIZE;
    // For each tile
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            const unsigned int tile_offset = (t_y * TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE); 
            // For each pixel within the tile
            for (unsigned int p_x = 0; p_x < TILE_SIZE; ++p_x) {
                for (unsigned int p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    // Load pixel
                    const unsigned int pixel_offset = (p_y * input_image->width + p_x); 
                    const unsigned char pixel = input_image->data[tile_offset + pixel_offset];
                    // Interpolate histogram values
                    unsigned char lerp_pixel;
                    // Decide how to interpolate based on the pixel position
                    // The branching could be removed, by making boundary interpolation interpolate against it's own tile with clamping
                    // Corners, no interpolation
                    if (((t_x == 0 && p_x < HALF_TILE_SIZE) || (t_x == TILES_X - 1 && p_x >= HALF_TILE_SIZE)) &&
                        ((t_y == 0 && p_y < HALF_TILE_SIZE) || (t_y == TILES_Y - 1 && p_y >= HALF_TILE_SIZE))) {
                        lerp_pixel = histograms[t_x][t_y].equalised_histogram[pixel];
                    // X Border, linear interpolation
                    } else if ((t_x == 0 && p_x < HALF_TILE_SIZE) || (t_x == TILES_X - 1 && p_x >= HALF_TILE_SIZE)) {
                        const int direction = lerp_direction(p_y);
                        const unsigned char home_pixel = histograms[t_x][t_y].equalised_histogram[pixel];
                        const unsigned char away_pixel = histograms[t_x][t_y + direction].equalised_histogram[pixel];
                        const float home_weight = lerp_weight(p_y);
                        lerp_pixel = mix_uc(home_pixel, away_pixel, home_weight);
                    // Y Border, linear interpolation
                    } else if ((t_y == 0 && p_y < HALF_TILE_SIZE) || (t_y == TILES_Y - 1 && p_y >= HALF_TILE_SIZE)) {
                        const int direction = lerp_direction(p_x);
                        const unsigned char home_pixel = histograms[t_x][t_y].equalised_histogram[pixel];
                        const unsigned char away_pixel = histograms[t_x + direction][t_y].equalised_histogram[pixel];
                        const float home_weight = lerp_weight(p_x);
                        lerp_pixel = mix_uc(home_pixel, away_pixel, home_weight);
                    // Centre, bilinear interpolation
                    } else {
                        const int direction_x = lerp_direction(p_x);
                        const int direction_y = lerp_direction(p_y);
                        // Lerp home row
                        float home_lerp;
                        {
                            const unsigned char home_pixel = histograms[t_x][t_y].equalised_histogram[pixel];
                            const unsigned char away_pixel = histograms[t_x + direction_x][t_y].equalised_histogram[pixel];
                            const float home_weight = lerp_weight(p_x);
                            home_lerp = mix_uc(home_pixel, away_pixel, home_weight);
                        }
                        // Lerp away row
                        float away_lerp;
                        {
                            const unsigned char home_pixel = histograms[t_x][t_y + direction_y].equalised_histogram[pixel];
                            const unsigned char away_pixel = histograms[t_x + direction_x][t_y + direction_y].equalised_histogram[pixel];
                            const float home_weight = lerp_weight(p_x);
                            away_lerp = mix_uc(home_pixel, away_pixel, home_weight);
                        }
                        // Lerp home and away over column
                        {
                            const float home_weight = lerp_weight(p_y);
                            lerp_pixel = (unsigned char)mix_f(home_lerp, away_lerp, home_weight);
                        }
                    }
                    // Store pixel
                    output_image->data[tile_offset + pixel_offset] = lerp_pixel;
                }
            }
        }
    }
    skip_interpolate_used++;
}

void reverse_algorithm(const Image *input_image, Image *output_image) {
    const unsigned int TILES_X = input_image->width / TILE_SIZE;
    const unsigned int TILES_Y = input_image->height / TILE_SIZE;
    // Allocate histogram
    Histogram **histograms = (Histogram **)malloc(TILES_X * sizeof(Histogram*));
    histograms[0] = (Histogram *)malloc(TILES_X * TILES_Y * sizeof(Histogram));
    memset(histograms[0],0, TILES_X * TILES_Y * sizeof(Histogram));
    for (unsigned int t_x = 1; t_x < TILES_X; ++t_x) {
        histograms[t_x] = histograms[0] + t_x * TILES_Y;         
    }
    // Allocate output image
    output_image->width = input_image->width;
    output_image->height = input_image->height;
    output_image->data = (unsigned char*)malloc(input_image->width * input_image->height * sizeof(unsigned char));
    // Generate histogram
    skip_histogram(input_image, histograms);
    // For each histogram
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            // find the min and max contrast values
            unsigned int min_c = PIXEL_RANGE-1;
            for (int i = 0; i < PIXEL_RANGE; ++i) {
                if (histograms[t_x][t_y].histogram[i]) {
                    min_c = i;
                    break;
                }
            }
            // min_c correpsonds to contrast that was clipped by contrast limit, so return that
            unsigned int pixels_over = 0;
            for (int i = 0; i < PIXEL_RANGE; ++i) {
                histograms[t_x][t_y].limited_histogram[i] = histograms[t_x][t_y].histogram[i] - min_c;
                if (histograms[t_x][t_y].limited_histogram[i] >= ABSOLUTE_CONTRAST_LIMIT) {
                    pixels_over++;
                    break;
                }
            }
            const unsigned int limited_contrast = (min_c * PIXEL_RANGE)/pixels_over;
            for (int i = 0; i < PIXEL_RANGE; ++i) {
                if (histograms[t_x][t_y].limited_histogram[i] >= ABSOLUTE_CONTRAST_LIMIT) {
                    histograms[t_x][t_y].limited_histogram[i] += limited_contrast;
                    break;
                }
            }
        }
    }
    skip_cumulative_histogram(TILES_X, TILES_Y, histograms);
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            // Find min again
            unsigned int min_c = PIXEL_RANGE-1;
            for (int i = 0; i < PIXEL_RANGE; ++i) {
                if (histograms[t_x][t_y].histogram[i]) {
                    min_c = i;
                    break;
                }
            }
            // Calculate unequalised histogram value
            for (unsigned int i = 0; i < PIXEL_RANGE; ++i) {
                float t = roundf(((histograms[t_x][t_y].cumulative_histogram[i] + min_c) / (float)(TILE_PIXELS + (min_c * PIXEL_RANGE))) * (float)PIXEL_MAX);
                t = t > PIXEL_MAX ? PIXEL_MAX : t; // indices before cdf_min overflow
                // Clamp value to bounds
                histograms[t_x][t_y].equalised_histogram[i] = (unsigned char)t;  
            }
        }
    }
    skip_interpolate(input_image, histograms, output_image);
}

int getSkipUsed() {
    return skip_histogram_used + skip_limited_histogram_used + skip_cumulative_histogram_used + skip_equalised_histogram_used + skip_interpolate_used;
}
int getStage1SkipUsed() {
    return skip_histogram_used;
}
int getStage2SkipUsed() {
    return skip_limited_histogram_used + skip_cumulative_histogram_used + skip_equalised_histogram_used;
}
int getStage3SkipUsed() {
    return skip_interpolate_used;
}
