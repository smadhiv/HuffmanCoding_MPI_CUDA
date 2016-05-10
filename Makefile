all: mpiCompress.o CudaCompress.o
	mpicc -o CudaMPI/MPICudaCompress CudaCompress.o -L /opt/cuda-toolkit/5.5.22/lib64 -lcudart mpiCompress.o 

mpiCompress.o: MPI/mpiCompress.c
	mpicc -c MPI/mpiCompress.c

CudaCompress.o: GPU/CudaCompress.cu
	nvcc -c GPU/CudaCompress.cu -arch=sm_35

clean:
	rm -rf *.o CudaCompress mpiCompress
	
