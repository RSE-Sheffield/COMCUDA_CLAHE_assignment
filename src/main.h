#ifndef __main_h__
#define __main_h__

enum Mode{CPU, OPENMP, CUDA};
typedef enum Mode Mode;
/**
 * Structure containing the options provided by runtime arguments
 */
struct Config {
    /**
     * Path to input image
     */
    char *input_file;
    /**
     * Path to output image (must be .png)
     */
    char *output_file;
    /**
     * Which algorithm to use CPU, OpenMP, CUDA
     */
    Mode mode;
    /**
     * Treated as boolean, program will operate in benchmark mode
     * This repeats the algorithm multiple times and returns an average time
     * It may also warn about incorrect settings
     */
    unsigned char benchmark;
}; typedef struct Config Config;
/**
 * This structure represents a multi-channel image
 * It is used internally to create single channel images to pass to the algorithm
 */
struct CImage {
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
    /**
     * Number of colour channels, e.g. 1 (greyscale), 3 (rgb), 4 (rgba)
     */
    int channels;
};
typedef struct CImage CImage;
/**
 * This structure represents a multi-channel image
 * It is used internally to create single channel images to pass to the algorithm
 */
struct HSVImage {
   /**
    * Array of pixel data of the image, 1 unsigned char per pixel channel
    * Pixels ordered left to right, top to bottom
    * There is no stride, this is a compact storage
    */
    float *h; // Angle in degrees [0-360]
    float *s; //Fractional value [0-1]
    unsigned char *v; //Fractional value [0-255], divide by 255 for the real value
    // Optional alpha channel
    unsigned char *a; // Unchanged from input image
    /**
     * Image width and height
     */
    int width, height;
    /**
     * Number of colour channels, e.g. 1 (greyscale), 3 (rgb), 4 (rgba)
     */
    int channels;
};
struct Runtimes
{
    float init;
    float stage1;
    float stage2;
    float stage3;
    float cleanup;
    float total;
};
typedef struct HSVImage HSVImage;
/**
 * Parse the runtime args into config
 * @param argc argc from main()
 * @param argv argv from main()]
 * @param config Pointer to config structure for return value
 */
void parse_args(int argc, char **argv, Config *config);
/**
 * Print runtime args and exit
 * @param program_name argv[0] should always be passed to this parameter
 */
void print_help(const char *program_name);
/**
 * Return the corresponding string for the provided Mode enum
 */
const char *mode_to_string(Mode m);

void rgb2hsv(const unsigned char src_r, const unsigned char src_g, const unsigned char src_b, float *dst_h, float *dst_s, unsigned char *dst_v);
void hsv2rgb(const float src_h, const float src_s, const unsigned char src_v, unsigned char *dst_r, unsigned char *dst_g, unsigned char *dst_b);
#endif  // __main_h__
