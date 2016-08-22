all:
	nvcc -dc GPUCompress.cu ../include/parallelFunctions.cu ../include/kernel.cu ../include/GPUWrapper.cu
	nvcc *.o -o ../bin/CUDA_compress
	rm -rf *.o
