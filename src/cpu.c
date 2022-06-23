#include "cpu.h"
#include "helper.h"

#include <stdlib.h>
#include <string.h>

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

///
/// Algorithm storage
///
Histogram_uint* cpu_histograms;
Histogram_uint* cpu_limited_histograms;
Histogram_uint* cpu_cumulative_histograms;
Histogram_uchar* cpu_equalised_histograms;
Image cpu_input_image;
Image cpu_output_image;
unsigned int cpu_TILES_X, cpu_TILES_Y;

///
/// Implementation
///
void cpu_begin(const Image *input_image) {
    cpu_TILES_X = input_image->width / TILE_SIZE;
    cpu_TILES_Y = input_image->height / TILE_SIZE;

    // Allocate histogram per tile
    cpu_histograms = (Histogram_uint*)malloc(cpu_TILES_X * cpu_TILES_Y * sizeof(Histogram_uint));
    cpu_limited_histograms = (Histogram_uint*)malloc(cpu_TILES_X * cpu_TILES_Y * sizeof(Histogram_uint));
    cpu_cumulative_histograms = (Histogram_uint*)malloc(cpu_TILES_X * cpu_TILES_Y * sizeof(Histogram_uint));
    cpu_equalised_histograms = (Histogram_uchar*)malloc(cpu_TILES_X * cpu_TILES_Y * sizeof(Histogram_uchar));

    // Zero all the histograms
    memset(cpu_histograms, 0, cpu_TILES_X * cpu_TILES_Y * sizeof(Histogram_uint));
    memset(cpu_limited_histograms, 0, cpu_TILES_X * cpu_TILES_Y * sizeof(Histogram_uint));
    memset(cpu_cumulative_histograms, 0, cpu_TILES_X * cpu_TILES_Y * sizeof(Histogram_uint));
    memset(cpu_equalised_histograms, 0, cpu_TILES_X * cpu_TILES_Y * sizeof(Histogram_uchar));
    
    // Allocate copy of input image
    cpu_input_image = *input_image;
    cpu_input_image.data = (unsigned char *)malloc(input_image->width * input_image->height);
    memcpy(cpu_input_image.data, input_image->data, input_image->width * input_image->height);

    // Allocate output image
    cpu_output_image = *input_image;
    cpu_output_image.data = (unsigned char *)malloc(input_image->width * input_image->height);

}
int cpu_stage1() {
    unsigned long long global_histogram[PIXEL_RANGE];
    memset(global_histogram, 0, sizeof(unsigned long long) * PIXEL_RANGE);
    // Generate histogram per tile
    for (unsigned int t_x = 0; t_x < cpu_TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < cpu_TILES_Y; ++t_y) {
            const unsigned int tile_index = (t_y * cpu_TILES_X + t_x);
            const unsigned int tile_offset = (t_y * cpu_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE); 
            // For each pixel within the tile
            for (int p_x = 0; p_x < TILE_SIZE; ++p_x) {
                for (int p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    // Load pixel
                    const unsigned int pixel_offset = (p_y * cpu_input_image.width + p_x); 
                    const unsigned char pixel = cpu_input_image.data[tile_offset + pixel_offset];
                    cpu_histograms[tile_index].histogram[pixel]++;
                    global_histogram[pixel]++;
                }
            }
        }
    }
    // Find the most common contrast value
    unsigned long long max_c = 0;  // Max count of pixels with a specific contrast value
    int max_i = -1; // Index (contrast value) of the histogram bin with max_c, init with an invalid value
    for (int i = 0; i < PIXEL_RANGE; ++i) {
        if (max_c < global_histogram[i]) {
            max_c = global_histogram[i];
            max_i = i;
        }
    }
#ifdef VALIDATION
    validate_histogram(&cpu_input_image, cpu_histograms, max_i);
#endif
    // Return the contrast value (it's index in the histogram), not the number of occurrences!
    return max_i;
}
void cpu_stage2() {
    // Normalise histograms
    // https://en.wikipedia.org/wiki/Histogram_equalization#Examples
    {
        // For each histogram
        for (unsigned int t_x = 0; t_x < cpu_TILES_X; ++t_x) {
            for (unsigned int t_y = 0; t_y < cpu_TILES_Y; ++t_y) {
                const unsigned int tile_index = (t_y * cpu_TILES_X + t_x);
                // Clamp where count exceeds ABSOLUTE_CONTRAST_LIMIT
                int extra_contrast = 0;
                for (unsigned int i = 0; i < PIXEL_RANGE; ++i) {
                    if (cpu_histograms[tile_index].histogram[i] > ABSOLUTE_CONTRAST_LIMIT) {
                        extra_contrast += cpu_histograms[tile_index].histogram[i] - ABSOLUTE_CONTRAST_LIMIT;
                        cpu_limited_histograms[tile_index].histogram[i] = ABSOLUTE_CONTRAST_LIMIT;
                    } else {
                        cpu_limited_histograms[tile_index].histogram[i] = cpu_histograms[tile_index].histogram[i];
                    }
                }
                const int bonus_contrast = extra_contrast / PIXEL_RANGE;  // integer division is fine here
                const int lost_contrast = extra_contrast % PIXEL_RANGE;
                if (bonus_contrast) {
                    for (int i = 0; i < PIXEL_RANGE; ++i) {
                        cpu_limited_histograms[tile_index].histogram[i] += bonus_contrast;
                    }
                }
                // Find cdf_min and convert histogram to cumulative
                // This is essentially a scan
                cpu_cumulative_histograms[tile_index].histogram[0] = cpu_limited_histograms[tile_index].histogram[0];
                unsigned int cdf_min = cpu_cumulative_histograms[tile_index].histogram[0];
                for (unsigned int i = 1; i < PIXEL_RANGE; ++i) {
                    cpu_cumulative_histograms[tile_index].histogram[i] = cpu_cumulative_histograms[tile_index].histogram[i-1] + cpu_limited_histograms[tile_index].histogram[i];
                    if (cpu_cumulative_histograms[tile_index].histogram[i-1] == 0 && cpu_cumulative_histograms[tile_index].histogram[i] != 0) { // Second half of condition is redundant in serial
                        cdf_min = cpu_cumulative_histograms[tile_index].histogram[i];
                    }
                }
                // Calculate equalised histogram value
                for (unsigned int i = 0; i < PIXEL_RANGE; ++i) {
                    float t = roundf(((cpu_cumulative_histograms[tile_index].histogram[i] - cdf_min) / (float)(TILE_PIXELS - lost_contrast)) * (float)(PIXEL_RANGE - 2)) + 1.0f;
                    t = t > PIXEL_MAX ? PIXEL_MAX : t; // indices before cdf_min overflow
                    // Clamp value to bounds
                    cpu_equalised_histograms[tile_index].histogram[i] = (unsigned char)t;
                    
                }
            }
        }
    }
#ifdef VALIDATION
    // validate_limited_histogram(cpu_TILES_X, cpu_TILES_Y, cpu_histograms, cpu_limited_histograms);
    // validate_cumulative_histogram(cpu_TILES_X, cpu_TILES_Y, cpu_limited_histograms, cpu_cumulative_histograms);
    validate_equalised_histogram(cpu_TILES_X, cpu_TILES_Y, cpu_histograms, cpu_equalised_histograms);
#endif
}
void cpu_stage3() {
    // Calculate interpolated pixels from normalised histograms
    {
        // For each tile
        for (unsigned int t_x = 0; t_x < cpu_TILES_X; ++t_x) {
            for (unsigned int t_y = 0; t_y < cpu_TILES_Y; ++t_y) {
                const unsigned int tile_offset = (t_y * cpu_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE); 
                // For each pixel within the tile
                for (unsigned int p_x = 0; p_x < TILE_SIZE; ++p_x) {
                    for (unsigned int p_y = 0; p_y < TILE_SIZE; ++p_y) {
                        const unsigned int tile_index = (t_y * cpu_TILES_X + t_x);
                        // Load pixel
                        const unsigned int pixel_offset = (p_y * cpu_input_image.width + p_x); 
                        const unsigned char pixel = cpu_input_image.data[tile_offset + pixel_offset];
                        // Interpolate histogram values
                        unsigned char lerp_pixel;
                        // Decide how to interpolate based on the pixel position
                        // The branching could be removed, by making boundary interpolation interpolate against it's own tile with clamping
                        // Corners, no interpolation
                        if (((t_x == 0 && p_x < HALF_TILE_SIZE) || (t_x == cpu_TILES_X - 1 && p_x >= HALF_TILE_SIZE)) &&
                            ((t_y == 0 && p_y < HALF_TILE_SIZE) || (t_y == cpu_TILES_Y - 1 && p_y >= HALF_TILE_SIZE))) {
                            lerp_pixel = cpu_equalised_histograms[tile_index].histogram[pixel];
                        // X Border, linear interpolation
                        } else if ((t_x == 0 && p_x < HALF_TILE_SIZE) || (t_x == cpu_TILES_X - 1 && p_x >= HALF_TILE_SIZE)) {
                            const int direction = lerp_direction(p_y);
                            const unsigned int tile_index_away = ((t_y + direction) * cpu_TILES_X + t_x);
                            const unsigned char home_pixel = cpu_equalised_histograms[tile_index].histogram[pixel];
                            const unsigned char away_pixel = cpu_equalised_histograms[tile_index_away].histogram[pixel];
                            const float home_weight = lerp_weight(p_y);
                            lerp_pixel = mix_uc(home_pixel, away_pixel, home_weight);
                        // Y Border, linear interpolation
                        } else if ((t_y == 0 && p_y < HALF_TILE_SIZE) || (t_y == cpu_TILES_Y - 1 && p_y >= HALF_TILE_SIZE)) {
                            const int direction = lerp_direction(p_x);
                            const unsigned int tile_index_away = (t_y * cpu_TILES_X + (t_x + direction));
                            const unsigned char home_pixel = cpu_equalised_histograms[tile_index].histogram[pixel];
                            const unsigned char away_pixel = cpu_equalised_histograms[tile_index_away].histogram[pixel];
                            const float home_weight = lerp_weight(p_x);
                            lerp_pixel = mix_uc(home_pixel, away_pixel, home_weight);
                        // Centre, bilinear interpolation
                        } else {
                            const int direction_x = lerp_direction(p_x);
                            const int direction_y = lerp_direction(p_y);
                            // Lerp home row
                            float home_lerp;
                            {
                                const unsigned int tile_index_away = (t_y * cpu_TILES_X + (t_x + direction_x));
                                const unsigned char home_pixel = cpu_equalised_histograms[tile_index].histogram[pixel];
                                const unsigned char away_pixel = cpu_equalised_histograms[tile_index_away].histogram[pixel];
                                const float home_weight = lerp_weight(p_x);
                                home_lerp = mix_uc(home_pixel, away_pixel, home_weight);
                            }
                            // Lerp away row
                            float away_lerp;
                            {
                                const unsigned int tile_index_home = ((t_y + direction_y) * cpu_TILES_X + t_x);
                                const unsigned int tile_index_away = ((t_y + direction_y) * cpu_TILES_X + (t_x + direction_x));
                                const unsigned char home_pixel = cpu_equalised_histograms[tile_index_home].histogram[pixel];
                                const unsigned char away_pixel = cpu_equalised_histograms[tile_index_away].histogram[pixel];
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
                        cpu_output_image.data[tile_offset + pixel_offset] = lerp_pixel;
                    }
                }
            }
        }
    }
#ifdef VALIDATION
    validate_interpolate(&cpu_input_image, cpu_equalised_histograms, &cpu_output_image);
#endif
}
void cpu_end(Image *output_image) {
    // Store return value
    output_image->width = cpu_output_image.width;
    output_image->height = cpu_output_image.height;
    memcpy(output_image->data, cpu_output_image.data, output_image->width * output_image->height * sizeof(unsigned char));
    // Release allocations
    free(cpu_output_image.data);
    free(cpu_input_image.data);
    free(cpu_histograms);
    free(cpu_limited_histograms);
    free(cpu_cumulative_histograms);
    free(cpu_equalised_histograms);
    // Return ptrs to nullptr
    cpu_output_image.data = 0;
    cpu_input_image.data = 0;
    cpu_histograms = 0;
    cpu_limited_histograms = 0;
    cpu_cumulative_histograms = 0;
    cpu_equalised_histograms = 0;
}
