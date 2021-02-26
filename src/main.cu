#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"

#include "main.h"
#include "config.h"
#include "common.h"
#include "cpu.h"
#include "openmp.h"
#include "cuda.cuh"
#include "helper.h"

int main(int argc, char **argv)
{
    // Parse args
    Config config;
    parse_args(argc, argv, &config);

    // Load image
    CImage user_cimage;
    {
        user_cimage.data = stbi_load(config.input_file, &user_cimage.width, &user_cimage.height, &user_cimage.channels, 0);
        if (!user_cimage.data) {
            printf("Unable to load image '%s', please try a different file.\n", config.input_file);
            return EXIT_FAILURE;
        }
        if (user_cimage.channels == 2) {
            printf("2 channel images are not supported, please try a different file.\n");
            return EXIT_FAILURE;
        }
    }

    // Format image (e.g. crop to multiple of block size)
    CImage input_cimage;
    {
        input_cimage.width = (user_cimage.width / TILE_SIZE) * TILE_SIZE;
        input_cimage.height = (user_cimage.height / TILE_SIZE) * TILE_SIZE;
        input_cimage.channels = user_cimage.channels;
        input_cimage.data = (unsigned char *)malloc(input_cimage.width * input_cimage.height * input_cimage.channels * sizeof(unsigned char));
        const int user_row_width = user_cimage.width * user_cimage.channels;
        const int input_row_width = input_cimage.width * input_cimage.channels;
        // Copy cropped data across
        for (int y = 0; y < input_cimage.height; ++y) {
            memcpy(input_cimage.data + y*input_row_width, user_cimage.data + y*user_row_width, input_row_width);
        }
    }
    stbi_image_free(user_cimage.data);

    // Export cleaned input image
    {
        if (!stbi_write_png("input.png", input_cimage.width, input_cimage.height, input_cimage.channels, input_cimage.data, input_cimage.width * input_cimage.channels)) {
            printf("Unable to save clean input image to input.png.\n");
            // return EXIT_FAILURE;
        }
    }

    // Convert image to HSV
    HSVImage hsv_image;
    {
        // Copy metadata
        hsv_image.width = input_cimage.width;
        hsv_image.height = input_cimage.height;
        hsv_image.channels = input_cimage.channels;
        // Allocate memory
        hsv_image.h = (float *)malloc(input_cimage.width * input_cimage.height * sizeof(float));
        hsv_image.s = (float *)malloc(input_cimage.width * input_cimage.height * sizeof(float));
        hsv_image.v = (unsigned char *)malloc(input_cimage.width * input_cimage.height * sizeof(unsigned char));
        hsv_image.a = (unsigned char *)malloc(input_cimage.width * input_cimage.height * sizeof(unsigned char));
        if (input_cimage.channels >= 3) {
            // Copy and convert data
            for (int i = 0; i < hsv_image.width * hsv_image.height; ++i) {
                rgb2hsv(
                input_cimage.data[(i * input_cimage.channels) + 0],
                input_cimage.data[(i * input_cimage.channels) + 1],
                input_cimage.data[(i * input_cimage.channels) + 2],
                hsv_image.h + i,
                hsv_image.s + i,
                hsv_image.v + i);
                if (input_cimage.channels == 4) hsv_image.a[i] = input_cimage.data[(i * input_cimage.channels) + 3];
            }
        } else if (input_cimage.channels == 1) {
            // Single channel can just be dumped into v
            memcpy(hsv_image.v, input_cimage.data, input_cimage.width * input_cimage.height * sizeof(unsigned char));
        }
    }
    free(input_cimage.data);

    // Create single channeled image from the HSV to pass to algorithm
    Image input_image;
    {
        // Copy metadata
        input_image.width = hsv_image.width;
        input_image.height = hsv_image.height;
        // Allocate memory
        input_image.data = (unsigned char *)malloc(input_image.width * input_image.height * sizeof(unsigned char));
        //Copy data
        memcpy(input_image.data, hsv_image.v, input_image.width * input_image.height * sizeof(unsigned char));
    }

    // Create result for validation
    Image validation_image;
    // Location to store result of stage1
    int validation_most_common_contrast[PIXEL_RANGE];
    {
        // Copy metadata
        validation_image.width = input_image.width;
        validation_image.height = input_image.height;
        // Allocate memory
        validation_image.data = (unsigned char *)malloc(input_image.width * input_image.height * sizeof(unsigned char));
        const unsigned int TILES_X = validation_image.width / TILE_SIZE;
        const unsigned int TILES_Y = validation_image.height / TILE_SIZE;
        // Allocate  histogram
        Histogram **histograms = (Histogram **)malloc(TILES_X * sizeof(Histogram*));
        histograms[0] = (Histogram *)malloc(TILES_X * TILES_Y * sizeof(Histogram));
        memset(histograms[0],0, TILES_X * TILES_Y * sizeof(Histogram));
        for (unsigned int t_x = 1; t_x < TILES_X; ++t_x) {
            histograms[t_x] = histograms[0] + t_x * TILES_Y;         
        }
        // Run algorithm
        skip_histogram(&input_image, histograms);
        skip_limited_histogram(TILES_X, TILES_Y, histograms);
        skip_cumulative_histogram(TILES_X, TILES_Y, histograms);
        skip_equalised_histogram(TILES_X, TILES_Y, histograms);
        skip_interpolate(&input_image, histograms, &validation_image);
        // Find all contrast values with max (small chance multiple contrast values share max)
        {
            for (int i = 0; i < PIXEL_RANGE; ++i)
                validation_most_common_contrast[i] = -1;
            unsigned long long global_histogram[PIXEL_RANGE];
            memset(global_histogram, 0, sizeof(unsigned long long) * PIXEL_RANGE);
            // Generate histogram per tile
            for (unsigned int i = 0; i < (unsigned int)(input_image.width * input_image.height); ++i) {                
                const unsigned char pixel = input_image.data[i];
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
        // Free temporary resources
        free(histograms[0]);
        free(histograms);
    }
       
    Image output_image;
    Runtimes timing_log;
    // Location to store result of stage1
    int most_common_contrast = -1;
    const int TOTAL_RUNS = config.benchmark ? BENCHMARK_RUNS : 1;
    {
        //Init for run  
        cudaEvent_t startT, initT, stage1T, stage2T, stage3T, stopT;
        CUDA_CALL(cudaEventCreate(&startT));
        CUDA_CALL(cudaEventCreate(&initT));
        CUDA_CALL(cudaEventCreate(&stage1T));
        CUDA_CALL(cudaEventCreate(&stage2T));
        CUDA_CALL(cudaEventCreate(&stage3T));
        CUDA_CALL(cudaEventCreate(&stopT));

        // Run 1 or many times
        memset(&timing_log, 0, sizeof(Runtimes));
        for (int runs = 0; runs < TOTAL_RUNS; ++runs) {
            if (TOTAL_RUNS > 1)
                printf("\r%d/%d", runs + 1, TOTAL_RUNS);
            memset(&output_image, 0, sizeof(Image));
            output_image.data = (unsigned char*)malloc(input_image.width * input_image.height);
            // Run Adaptive Histogram algorithm
            CUDA_CALL(cudaEventRecord(startT));
            CUDA_CALL(cudaEventSynchronize(startT));
            switch (config.mode) {
            case CPU:
                {
                    cpu_begin(&input_image);
                    CUDA_CALL(cudaEventRecord(initT));
                    CUDA_CALL(cudaEventSynchronize(initT));
                    most_common_contrast = cpu_stage1();
                    CUDA_CALL(cudaEventRecord(stage1T));
                    CUDA_CALL(cudaEventSynchronize(stage1T));
                    cpu_stage2();
                    CUDA_CALL(cudaEventRecord(stage2T));
                    CUDA_CALL(cudaEventSynchronize(stage2T));
                    cpu_stage3();
                    CUDA_CALL(cudaEventRecord(stage3T));
                    CUDA_CALL(cudaEventSynchronize(stage3T));
                    cpu_end(&output_image);
                }
                break;
            case OPENMP:
                {
                    openmp_begin(&input_image);
                    CUDA_CALL(cudaEventRecord(initT));
                    CUDA_CALL(cudaEventSynchronize(initT));
                    most_common_contrast = openmp_stage1();
                    CUDA_CALL(cudaEventRecord(stage1T));
                    CUDA_CALL(cudaEventSynchronize(stage1T));
                    openmp_stage2();
                    CUDA_CALL(cudaEventRecord(stage2T));
                    CUDA_CALL(cudaEventSynchronize(stage2T));
                    openmp_stage3();
                    CUDA_CALL(cudaEventRecord(stage3T));
                    CUDA_CALL(cudaEventSynchronize(stage3T));
                    openmp_end(&output_image);
                }
                break;
            case CUDA:
                {
                    cuda_begin(&input_image);
                    CUDA_CHECK("cuda_begin()");
                    CUDA_CALL(cudaEventRecord(initT));
                    CUDA_CALL(cudaEventSynchronize(initT));
                    most_common_contrast = cuda_stage1();
                    CUDA_CHECK("cuda_stage1()");
                    CUDA_CALL(cudaEventRecord(stage1T));
                    CUDA_CALL(cudaEventSynchronize(stage1T));
                    cuda_stage2();
                    CUDA_CHECK("cuda_stage2()");
                    CUDA_CALL(cudaEventRecord(stage2T));
                    CUDA_CALL(cudaEventSynchronize(stage2T));
                    cuda_stage3();
                    CUDA_CHECK("cuda_stage3()");
                    CUDA_CALL(cudaEventRecord(stage3T));
                    CUDA_CALL(cudaEventSynchronize(stage3T));
                    cuda_end(&output_image);
                }
                break;
            }
            CUDA_CALL(cudaEventRecord(stopT));
            CUDA_CALL(cudaEventSynchronize(stopT));
            // Sum timing info
            float milliseconds = 0;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, startT, initT));
            timing_log.init += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, initT, stage1T));
            timing_log.stage1 += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, stage1T, stage2T));
            timing_log.stage2 += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, stage2T, stage3T));
            timing_log.stage3 += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, stage3T, stopT));
            timing_log.cleanup += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, startT, stopT));
            timing_log.total += milliseconds;
            // Avoid memory leak
            if (runs + 1 < TOTAL_RUNS) {
                if (output_image.data)
                    free(output_image.data);
            }
        }
        // Convert timing info to average
        timing_log.init /= TOTAL_RUNS;
        timing_log.stage1 /= TOTAL_RUNS;
        timing_log.stage2 /= TOTAL_RUNS;
        timing_log.stage3 /= TOTAL_RUNS;
        timing_log.cleanup /= TOTAL_RUNS;
        timing_log.total /= TOTAL_RUNS;

        // Cleanup timing
        cudaEventDestroy(startT);
        cudaEventDestroy(initT);
        cudaEventDestroy(stage1T);
        cudaEventDestroy(stage2T);
        cudaEventDestroy(stage3T);
        cudaEventDestroy(stopT);
    }

    // Copy output image v channel back to hsv
    if (output_image.data) {
        memcpy(hsv_image.v, output_image.data, input_image.width * input_image.height * sizeof(unsigned char));
    }

    // Convert HSV image back to rgb
    CImage output_cimage;
    {
        output_cimage.width = (hsv_image.width / TILE_SIZE) * TILE_SIZE;
        output_cimage.height = (hsv_image.height / TILE_SIZE) * TILE_SIZE;
        output_cimage.channels = hsv_image.channels;
        output_cimage.data = (unsigned char *)malloc(hsv_image.width * hsv_image.height * hsv_image.channels * sizeof(unsigned char));
        // Copy and convert data
        if (hsv_image.channels >= 3) {
            for (int i = 0; i < output_cimage.width * output_cimage.height; ++i) {
                hsv2rgb(
                    hsv_image.h[i],
                    hsv_image.s[i],
                    hsv_image.v[i],
                    output_cimage.data + (i * output_cimage.channels) + 0,
                    output_cimage.data + (i * output_cimage.channels) + 1,
                    output_cimage.data + (i * output_cimage.channels) + 2);
                if (output_cimage.channels == 4) output_cimage.data[(i * output_cimage.channels) + 3] = hsv_image.a[i];
            }
        } else if (hsv_image.channels == 1) {
            memcpy(output_cimage.data, hsv_image.v, hsv_image.width * hsv_image.height * sizeof(unsigned char));
        }
    }
    

    // Validate and report    
    {
        printf("\rValidation Status: \n");
        printf("\tImage width: %s\n", validation_image.width == output_image.width ? "Pass" : "Fail");
        printf("\tImage height: %s\n", validation_image.height == output_image.height ? "Pass" : "Fail");
        int v_size = validation_image.width * validation_image.height;
        int o_size = output_image.width * output_image.height;
        int s_size = v_size < o_size ? v_size : o_size;
        int bad_pixels = 0;
        int close_pixels = 0;
        if (output_image.data) {
            for (int i = 0; i < s_size; ++i) {
                if (output_image.data[i] != validation_image.data[i]) {
                    // Give a +-1 threshold for error (incase fast-math triggers a small difference in places)
                    if (output_image.data[i]+1 == validation_image.data[i] || output_image.data[i]-1 == validation_image.data[i]) {
                        close_pixels++;
                    } else {
                        bad_pixels++;
                    }
                }
            }
            printf("\tImage pixels: %s (%d/%u wrong)\n", bad_pixels ?  "Fail": "Pass", bad_pixels, o_size);
        } else {
            printf("\tImage pixels: Fail, (output_image->data not set)\n");
        }
        int bad_contrast = 1;
        for (int i = 0; i < PIXEL_RANGE && validation_most_common_contrast[i] != -1; ++i){
            if (most_common_contrast == validation_most_common_contrast[i]) {
                bad_contrast = 0;
                break;
            }
        }
        printf("\tMost common contrast value: %s\n", bad_contrast ? "Fail": "Pass");
    }

    // Export output image
    if (config.output_file) {
        if (!stbi_write_png(config.output_file, output_cimage.width, output_cimage.height, output_cimage.channels, output_cimage.data, output_cimage.width * output_cimage.channels)) {
            printf("Unable to save image output to %s.\n", config.output_file);
            // return EXIT_FAILURE;
        }
    }


    // Report timing information    
    printf("%s Average execution timing from %d runs\n", mode_to_string(config.mode), TOTAL_RUNS);
    if (config.mode == CUDA) {
        int device_id = 0;
        CUDA_CALL(cudaGetDevice(&device_id));
        cudaDeviceProp props;
        memset(&props, 0, sizeof(cudaDeviceProp));
        CUDA_CALL(cudaGetDeviceProperties(&props, device_id));
        printf("Using GPU: %s\n", props.name);
    }
#ifdef _DEBUG
    printf("Code built as DEBUG, timing results are invalid!\n");
#endif
    printf("Init: %.3fms\n", timing_log.init);
    printf("Stage 1: %.3fms%s\n", timing_log.stage1, getStage1SkipUsed() ? " (helper method used, time invalid)" : "");
    printf("Stage 2: %.3fms%s\n", timing_log.stage2, getStage2SkipUsed() ? " (helper method used, time invalid)" : "");
    printf("Stage 3: %.3fms%s\n", timing_log.stage3, getStage3SkipUsed() ? " (helper method used, time invalid)" : "");
    printf("Free: %.3fms\n", timing_log.cleanup);
    printf("Total: %.3fms%s\n", timing_log.total, getSkipUsed() ? " (helper method used, time invalid)" : "");

    // Cleanup
    cudaDeviceReset();
    free(validation_image.data);
    free(output_cimage.data);
    free(output_image.data);
    free(input_image.data);
    if (hsv_image.a)
        free(hsv_image.a);
    free(hsv_image.v);
    free(hsv_image.s);
    free(hsv_image.h);
    free(config.input_file);
    if (config.output_file)
        free(config.output_file);
    return EXIT_SUCCESS;
}
void parse_args(int argc, char **argv, Config *config) {
    // Clear config struct
    memset(config, 0, sizeof(Config));
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "Program expects 2-4 arguments, only %d provided.\n", argc-1);
        print_help(argv[0]);
    }
    // Parse first arg as mode
    {
        char lower_arg[7];  // We only care about first 6 characters
        // Convert to lower case
        int i = 0;
        for(; argv[1][i] && i < 6; i++){
            lower_arg[i] = tolower(argv[1][i]);
        }
        lower_arg[i] = '\0';
        // Check for a match
        if (!strcmp(lower_arg, "cpu")) {
            config->mode = CPU;
        } else if (!strcmp(lower_arg, "openmp")) {
            config->mode = OPENMP;
        } else if (!strcmp(lower_arg, "cuda") || !strcmp(lower_arg, "gpu")) {
            config->mode = CUDA;
        } else {
            fprintf(stderr, "Unexpected string provided as first argument: '%s' .\n", argv[1]);
            fprintf(stderr, "First argument expects a single mode as string: CPU, OPENMP, CUDA.\n");
            print_help(argv[0]);
        }
    }
    // Parse second arg as input file
    {
        // Find length of string
        const size_t input_name_len = strlen(argv[2]) + 1;  // Add 1 for null terminating character
        // Allocate memory and copy
        config->input_file = (char*)malloc(input_name_len);
        memcpy(config->input_file, argv[2], input_name_len);
    }
    
    // Iterate over remaining args    
    int i = 3;
    char * t_arg = 0;
    for (; i < argc; i++) {
        // Make a lowercase copy of the argument
        const size_t arg_len = strlen(argv[i]) + 1;  // Add 1 for null terminating character
        if (t_arg) 
            free(t_arg);
        t_arg = (char*)malloc(arg_len);
        int j = 0;
        for(; argv[i][j]; ++j){
            t_arg[j] = tolower(argv[i][j]);
        }
        t_arg[j] = '\0';
        // Decide which arg it is
        if (!strcmp("--bench", t_arg) || !strcmp("--benchmark", t_arg)|| !strcmp("-b", t_arg)) {
            config->benchmark = 1;
            continue;
        }
        if (!strcmp(t_arg + arg_len - 5, ".png")) {
            // Allocate memory and copy
            config->output_file = (char*)malloc(arg_len);
            memcpy(config->output_file, argv[i], arg_len);
            continue;
        }
        fprintf(stderr, "Unexpected optional argument: %s\n", argv[i]);
        print_help(argv[0]);
    }
    if (t_arg) 
        free(t_arg);
}
void print_help(const char *program_name) {
    fprintf(stderr, "%s <mode> <input image> (<output image>) (--bench)\n", program_name);
    
    const char *line_fmt = "%-18s %s\n";
    fprintf(stderr, "Required Arguments:\n");
    fprintf(stderr, line_fmt, "<mode>", "The algorithm to use: CPU, OPENMP, CUDA");
    fprintf(stderr, line_fmt, "<input image>", "Input image, .png, .jpg");
    fprintf(stderr, "Optional Arguments:\n");
    fprintf(stderr, line_fmt, "<output image>", "Output image, requires .png filetype");
    fprintf(stderr, line_fmt, "-b, --bench", "Enable benchmark mode");

    exit(EXIT_FAILURE);
}
const char *mode_to_string(Mode m) {
    switch (m)
    {
    case CPU:
      return "CPU";
    case OPENMP:
     return "OpenMP";
    case CUDA:
      return "CUDA";
    }
    return "?";
}
void rgb2hsv(const unsigned char src_r, const unsigned char src_g, const unsigned char src_b, float *dst_h, float *dst_s, unsigned char *dst_v)
{
    float r = src_r / 255.0f;
    float g = src_g / 255.0f;
    float b = src_b / 255.0f;

    float h, s, v; // h:0-360.0, s:0.0-1.0, v:0.0-1.0

    float max = r > g ? (r > b ? r : b) : (g > b ? g : b);
    float min = r < g ? (r < b ? r : b) : (g < b ? g : b);

    v = max;

    if (max == 0.0f) {
        s = 0;
        h = 0;
    }
    else if (max - min == 0.0f) {
        s = 0;
        h = 0;
    }
    else {
        s = (max - min) / max;

        if (max == r) {
            h = 60 * ((g - b) / (max - min)) + 0;
        }
        else if (max == g) {
            h = 60 * ((b - r) / (max - min)) + 120;
        }
        else {
            h = 60 * ((r - g) / (max - min)) + 240;
        }
    }

    if (h < 0) h += 360.0f;

    *dst_h = h;   // dst_h : 0-360
    *dst_s = s; // dst_s : 0-1
    *dst_v = (unsigned char)(v * 255); // dst_v : 0-255
}

void hsv2rgb(const float src_h, const float src_s, const unsigned char src_v, unsigned char *dst_r, unsigned char *dst_g, unsigned char *dst_b)
{
    float h = src_h; // 0-360
    float s = src_s; // 0.0-1.0
    float v = src_v / 255.0f; // 0.0-1.0

    float r, g, b; // 0.0-1.0

    int   hi = (int)(h / 60.0f) % 6;
    float f  = (h / 60.0f) - hi;
    float p  = v * (1.0f - s);
    float q  = v * (1.0f - s * f);
    float t  = v * (1.0f - s * (1.0f - f));

    switch(hi) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
        default: r=0, g=0, b=0;
    }

    *dst_r = (unsigned char)(r * 255); // dst_r : 0-255
    *dst_g = (unsigned char)(g * 255); // dst_r : 0-255
    *dst_b = (unsigned char)(b * 255); // dst_r : 0-255
}
