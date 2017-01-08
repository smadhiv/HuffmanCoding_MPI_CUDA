all:
	nvcc -dc CUDACompress.cu ../include/parallelFunctions.cu ../include/kernel.cu ../include/GPUWrapper.cu
	nvcc *.o -o ../bin/CUDA_compress
	rm -rf *.o

clean:
	if [ -a ../bin/CUDA_compress ]; then rm -f ../bin/CUDA_compress; fi;
