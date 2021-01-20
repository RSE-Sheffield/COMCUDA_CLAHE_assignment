# Change the example variable to build a different source module (e.g. hello/example1/example4)
EXAMPLE=AdaptiveHistogramOptimisation

# Makefile variables 
# Add extra targets to OBJ with space separator e.g. If there is as source file random.c then add random.o to OBJ)
# Add any additional dependencies (header files) to DEPS. e.g. if there is a header file random.h required by your source modules then add this to DEPS.
CC=gcc
CCFLAGS= -O3 -fopenmp -I. -Isrc
NVCC=nvcc
NVCC_FLAGS= -gencode arch=compute_35,code=compute_35 -I. -Isrc
OBJ=main.o cpu.o openmp.o cuda.o helper.o
DEPS=src/common.h src/config.h src/cpu.h src/cuda.cuh src/helper.h src/main.h src/openmp.h external/stb_image.h external/stb_image_write.h

# Build rule for object files ($@ is left hand side of rule, $< is first item from the right hand side of rule)
%.o : src/%.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS) $(addprefix -Xcompiler ,$(CCFLAGS))
  
%.o : src/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CCFLAGS)

# Make example ($^ is all items from right hand side of the rule)
$(EXAMPLE) : $(OBJ)
	$(NVCC) -o $@ $^ $(NVCC_FLAGS) $(addprefix -Xcompiler ,$(CCFLAGS))

# PHONY prevents make from doing something with a filename called clean
.PHONY : clean
clean:
	rm -rf $(EXAMPLE) $(OBJ)