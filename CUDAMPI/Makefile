all:
	nvcc -dc CUDAMPICompress.cu ../include/parallelFunctions.cu ../include/GPUWrapper.cu ../include/kernel.cu
	nvcc *.o -L/software/openmpi/1.10.3_gcc610/lib -lmpi -o ../bin/CUDAMPI_compress
	rm -f *.o

clean:
	if [ -a ../bin/CUDAMPI_compress ]; then rm -f ../bin/CUDAMPI_compress; fi;
