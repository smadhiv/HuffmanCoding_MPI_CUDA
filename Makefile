all: mpiCompress.o CudaCompress.o
	mpicc -o MPICudaCompress CudaCompress.o -L /opt/cuda-toolkit/5.5.22/lib64 -lcudart mpiCompress.o 

mpiCompress.o: mpiCompress.c
	mpicc -c mpiCompress.c

CudaCompress.o: CudaCompress.cu
	nvcc -c CudaCompress.cu

clean:
	rm -rf *.o CudaCompress mpiCompress
