all: mpiCompress.o CudaCompress.o
	mpicc -o bin/MPICudaCompress CudaCompress.o -L /opt/cuda-toolkit/5.5.22/lib64 -lcudart mpiCompress.o 
	rm -rf *.o

mpiCompress.o: MPI/mpiCompress.c
	mpicc -c MPI/mpiCompress.c

CudaCompress.o: GPU/CudaCompress.cu
	nvcc -c GPU/CudaCompress.cu -arch=sm_35

clean:
	rm -rf *.o
	rm bin/*
