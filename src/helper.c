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
void validate_histogram(const Image *input_image, Histogram_uint** test_histograms, int most_common_contrast) {
    const unsigned int TILES_X = input_image->width / TILE_SIZE;
    const unsigned int TILES_Y = input_image->height / TILE_SIZE;
    // Allocate and generate our own internal histogram
    Histogram_uint**histograms = (Histogram_uint**)malloc(TILES_X * sizeof(Histogram_uint*));
    histograms[0] = (Histogram_uint*)malloc(TILES_X * TILES_Y * sizeof(Histogram_uint));
    memset(histograms[0],0, TILES_X * TILES_Y * sizeof(Histogram_uint));
    for (unsigned int t_x = 1; t_x < TILES_X; ++t_x) {
        histograms[t_x] = histograms[0] + t_x * TILES_Y;         
    }
    skip_histogram(input_image, histograms);
    skip_histogram_used--;
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
    // Find all contrast values with max (small chance multiple contrast values share max)
    int validation_most_common_contrast[PIXEL_RANGE];
    {
        for (int i = 0; i < PIXEL_RANGE; ++i)
            validation_most_common_contrast[i] = -1;
        unsigned long long global_histogram[PIXEL_RANGE];
        memset(global_histogram, 0, sizeof(unsigned long long) * PIXEL_RANGE);
        // Generate histogram per tile
        for (unsigned int i = 0; i < (unsigned int)(input_image->width * input_image->height); ++i) {                
            const unsigned char pixel = input_image->data[i];
            global_histogram[pixel]++;
        }
        // Find max value
        // Find the most common contrast value
        unsigned long long max_c = 0;
        int max_i = 0;
        for (int i = 0; i < PIXEL_RANGE; ++i) {
            if (max_c < global_histogram[i]) {
                max_c = global_histogram[i];
                max_i = i;
            }
        }
        // Find everywhere it occurs
        int j = 0;
        for (int i = max_i; i < PIXEL_RANGE; ++i) {
            if (global_histogram[i] == max_c)
            validation_most_common_contrast[j++] = i;
        }
    }
    int bad_contrast = 1;
    for (int i = 0; i < PIXEL_RANGE && validation_most_common_contrast[i] != -1; ++i){
        if (most_common_contrast == validation_most_common_contrast[i]) {
            bad_contrast = 0;
            break;
        }
    }
    printf("validate_histogram() Most common contrast value: %s\n", bad_contrast ? "Fail": "Pass");
    
    // Release internal histogram
    free(histograms[0]);
    free(histograms);
}
int skip_histogram(const Image *input_image, Histogram_uint** histograms) {
    const unsigned int TILES_X = input_image->width / TILE_SIZE;
    const unsigned int TILES_Y = input_image->height / TILE_SIZE;
    unsigned long long global_histogram[PIXEL_RANGE];
    memset(global_histogram, 0, sizeof(unsigned long long) * PIXEL_RANGE);
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
                    global_histogram[pixel]++;
                }
            }
        }
    }
    // Find the most common contrast value
    unsigned long long max_c = 0;
    int max_i = -1; // Init with an invalid value
    for (int i = 0; i < PIXEL_RANGE; ++i) {
        max_c = max_c > global_histogram[i] ? max_c : global_histogram[i];
        max_i = i;
    }
    skip_histogram_used++;
    // Return the contrast value (it's index in the histogram), not the number of occurrences!
    return max_i;
}

int skip_limited_histogram_used = -1;
void validate_limited_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram_uint** histograms, Histogram_uint** test_limited_histograms) {
    // Allocate, copy and generate our own internal limited histogram
    Histogram_uint**limited_histograms = (Histogram_uint**)malloc(TILES_X * sizeof(Histogram_uint*));
    limited_histograms[0] = (Histogram_uint*)malloc(TILES_X * TILES_Y * sizeof(Histogram_uint));
    memset(limited_histograms[0],0, TILES_X * TILES_Y * sizeof(Histogram_uint));
    for (unsigned int t_x = 1; t_x < TILES_X; ++t_x) {
        limited_histograms[t_x] = limited_histograms[0] + t_x * TILES_Y;
    }
    skip_limited_histogram(TILES_X, TILES_Y, histograms, limited_histograms);
    skip_limited_histogram_used--;
    // Validate and report result
    unsigned int bad_histograms = 0;
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            for (unsigned int i = 0; i < PIXEL_RANGE; ++ i) {
                if (test_limited_histograms[t_x][t_y].histogram[i] != limited_histograms[t_x][t_y].histogram[i]) {
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
    free(limited_histograms[0]);
    free(limited_histograms);
}
void skip_limited_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram_uint** histograms, Histogram_uint** limited_histograms) {
    // For each histogram
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            // Clamp where count exceeds ABSOLUTE_CONTRAST_LIMIT
            int extra_contrast = 0;
            for (unsigned int i = 0; i < PIXEL_RANGE; ++i) {
                if (histograms[t_x][t_y].histogram[i] > ABSOLUTE_CONTRAST_LIMIT) {
                    extra_contrast += histograms[t_x][t_y].histogram[i] - ABSOLUTE_CONTRAST_LIMIT;
                    limited_histograms[t_x][t_y].histogram[i] = ABSOLUTE_CONTRAST_LIMIT;
                } else {
                    limited_histograms[t_x][t_y].histogram[i] = histograms[t_x][t_y].histogram[i];
                }
            }
            // int lost_contrast = 0;
            if (extra_contrast > PIXEL_RANGE) {
                const int bonus_contrast = extra_contrast / PIXEL_RANGE;  // integer division is fine here
                // lost_contrast = extra_contrast % PIXEL_RANGE;
                for (int i = 0; i < PIXEL_RANGE; ++i) {
                    limited_histograms[t_x][t_y].histogram[i] += bonus_contrast;
                }
            }
        }
    }
    skip_limited_histogram_used++;
}
int skip_cumulative_histogram_used = 0;
void validate_cumulative_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram_uint** limited_histograms, Histogram_uint** test_cumulative_histograms) {
    // Allocate, copy and generate our own internal histogram
    Histogram_uint** cumulative_histograms = (Histogram_uint**)malloc(TILES_X * sizeof(Histogram_uint*));
    cumulative_histograms[0] = (Histogram_uint*)malloc(TILES_X * TILES_Y * sizeof(Histogram_uint));
    memset(cumulative_histograms[0],0, TILES_X * TILES_Y * sizeof(Histogram_uint));
    for (unsigned int t_x = 1; t_x < TILES_X; ++t_x) {
        cumulative_histograms[t_x] = cumulative_histograms[0] + t_x * TILES_Y;
    }
    skip_cumulative_histogram(TILES_X, TILES_Y, limited_histograms, cumulative_histograms);
    skip_cumulative_histogram_used--;
    // Validate and report result
    unsigned int bad_histograms = 0;
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            for (unsigned int i = 0; i < PIXEL_RANGE; ++ i) {
                if (test_cumulative_histograms[t_x][t_y].histogram[i] != cumulative_histograms[t_x][t_y].histogram[i]) {
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
    free(cumulative_histograms[0]);
    free(cumulative_histograms);
}
void skip_cumulative_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram_uint** limited_histograms, Histogram_uint** cumulative_histograms) {
    // For each histogram
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            // Find cdf_min and convert histogram to cumulative
            // This is essentially a scan
            // unsigned int cdf_min = 0;
            cumulative_histograms[t_x][t_y].histogram[0] = limited_histograms[t_x][t_y].histogram[0];
            for (unsigned int i = 1; i < PIXEL_RANGE; ++i) {
                cumulative_histograms[t_x][t_y].histogram[i] = cumulative_histograms[t_x][t_y].histogram[i-1] + limited_histograms[t_x][t_y].histogram[i];
                // if (histograms[t_x][t_y].cumulative_histogram[i-1] == 0 && histograms[t_x][t_y].cumulative_histogram[i] != 0) { // Second half of condition is redundant in serial
                //     cdf_min = histograms[t_x][t_y].cumulative_histogram[i];
                // }
            }
        }
    }
    skip_cumulative_histogram_used++;
}
int skip_equalised_histogram_used = 0;
void validate_equalised_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram_uint** histograms, Histogram_uchar** test_equalisied_histograms) {
    // Allocate, copy and generate our own internal histogram
    Histogram_uchar** equalsied_histograms = (Histogram_uchar**)malloc(TILES_X * sizeof(Histogram_uchar*));
    equalsied_histograms[0] = (Histogram_uchar*)malloc(TILES_X * TILES_Y * sizeof(Histogram_uchar));
    memset(equalsied_histograms[0], 0, TILES_X* TILES_Y * sizeof(Histogram_uchar));
    for (unsigned int t_x = 1; t_x < TILES_X; ++t_x) {
        equalsied_histograms[t_x] = equalsied_histograms[0] + t_x * TILES_Y;
    }
    skip_equalised_histogram(TILES_X, TILES_Y, histograms, equalsied_histograms);
    skip_equalised_histogram_used--;
    // Validate and report result
    unsigned int bad_histograms = 0;
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            for (unsigned int i = 0; i < PIXEL_RANGE; ++ i) {
                if (test_equalisied_histograms[t_x][t_y].histogram[i] != equalsied_histograms[t_x][t_y].histogram[i]) {
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
    free(equalsied_histograms[0]);
    free(equalsied_histograms);
}
void skip_equalised_histogram(unsigned int TILES_X, unsigned int TILES_Y, Histogram_uint** histograms, Histogram_uchar** equalisied_histograms) {
    Histogram_uint** limited_histograms = (Histogram_uint**)malloc(TILES_X * sizeof(Histogram_uint*));
    limited_histograms[0] = (Histogram_uint*)malloc(TILES_X * TILES_Y * sizeof(Histogram_uint));
    memset(limited_histograms[0], 0, TILES_X * TILES_Y * sizeof(Histogram_uint));
    for (unsigned int t_x = 1; t_x < TILES_X; ++t_x) {
        limited_histograms[t_x] = limited_histograms[0] + t_x * TILES_Y;
    }
    Histogram_uint** cumulative_histograms = (Histogram_uint**)malloc(TILES_X * sizeof(Histogram_uint*));
    cumulative_histograms[0] = (Histogram_uint*)malloc(TILES_X * TILES_Y * sizeof(Histogram_uint));
    memset(cumulative_histograms[0], 0, TILES_X * TILES_Y * sizeof(Histogram_uint));
    for (unsigned int t_x = 1; t_x < TILES_X; ++t_x) {
        cumulative_histograms[t_x] = cumulative_histograms[0] + t_x * TILES_Y;
    }
    skip_limited_histogram(TILES_X, TILES_Y, histograms, limited_histograms);
    skip_limited_histogram_used--;
    skip_cumulative_histogram(TILES_X, TILES_Y, limited_histograms, cumulative_histograms);
    skip_cumulative_histogram_used--;
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
            // Find cdf_min (This requires cumulative histogram)
            unsigned int cdf_min = 0;
            for (unsigned int i = 1; i < PIXEL_RANGE; ++i) {
                if (cumulative_histograms[t_x][t_y].histogram[i-1] == 0 && cumulative_histograms[t_x][t_y].histogram[i] != 0) { // Second half of condition is redundant in serial
                    cdf_min = cumulative_histograms[t_x][t_y].histogram[i];
                    break;
                }
            }
            // Calculate equalised histogram value
            for (unsigned int i = 0; i < PIXEL_RANGE; ++i) {
                float t = roundf((((cumulative_histograms[t_x][t_y].histogram[i] - cdf_min) / (float)(TILE_PIXELS - lost_contrast)) * (float)(PIXEL_RANGE - 2)) + 1);
                t = t > PIXEL_MAX ? PIXEL_MAX : t; // indices before cdf_min overflow
                // Clamp value to bounds
                equalisied_histograms[t_x][t_y].histogram[i] = (unsigned char)t;
            }
        }
    }
    skip_equalised_histogram_used++;
    // Release internal histogram
    free(limited_histograms[0]);
    free(limited_histograms);
    free(cumulative_histograms[0]);
    free(cumulative_histograms);
}

int skip_interpolate_used = -1;
void validate_interpolate(const Image *input_image, Histogram_uchar** equalisied_histograms, Image *test_output_image) {
    // Allocate, copy and generate our own internal output image
    Image output_image;
    memcpy(&output_image, input_image, sizeof(Image));
    output_image.data =
        malloc(output_image.width * output_image.height * sizeof(unsigned char));
    
    skip_interpolate(input_image, equalisied_histograms, &output_image);
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
void skip_interpolate(const Image *input_image, Histogram_uchar** equalisied_histograms, Image *output_image) {
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
                        lerp_pixel = equalisied_histograms[t_x][t_y].histogram[pixel];
                    // X Border, linear interpolation
                    } else if ((t_x == 0 && p_x < HALF_TILE_SIZE) || (t_x == TILES_X - 1 && p_x >= HALF_TILE_SIZE)) {
                        const int direction = lerp_direction(p_y);
                        const unsigned char home_pixel = equalisied_histograms[t_x][t_y].histogram[pixel];
                        const unsigned char away_pixel = equalisied_histograms[t_x][t_y + direction].histogram[pixel];
                        const float home_weight = lerp_weight(p_y);
                        lerp_pixel = mix_uc(home_pixel, away_pixel, home_weight);
                    // Y Border, linear interpolation
                    } else if ((t_y == 0 && p_y < HALF_TILE_SIZE) || (t_y == TILES_Y - 1 && p_y >= HALF_TILE_SIZE)) {
                        const int direction = lerp_direction(p_x);
                        const unsigned char home_pixel = equalisied_histograms[t_x][t_y].histogram[pixel];
                        const unsigned char away_pixel = equalisied_histograms[t_x + direction][t_y].histogram[pixel];
                        const float home_weight = lerp_weight(p_x);
                        lerp_pixel = mix_uc(home_pixel, away_pixel, home_weight);
                    // Centre, bilinear interpolation
                    } else {
                        const int direction_x = lerp_direction(p_x);
                        const int direction_y = lerp_direction(p_y);
                        // Lerp home row
                        float home_lerp;
                        {
                            const unsigned char home_pixel = equalisied_histograms[t_x][t_y].histogram[pixel];
                            const unsigned char away_pixel = equalisied_histograms[t_x + direction_x][t_y].histogram[pixel];
                            const float home_weight = lerp_weight(p_x);
                            home_lerp = mix_uc(home_pixel, away_pixel, home_weight);
                        }
                        // Lerp away row
                        float away_lerp;
                        {
                            const unsigned char home_pixel = equalisied_histograms[t_x][t_y + direction_y].histogram[pixel];
                            const unsigned char away_pixel = equalisied_histograms[t_x + direction_x][t_y + direction_y].histogram[pixel];
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
