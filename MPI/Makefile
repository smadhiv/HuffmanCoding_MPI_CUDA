all:
	mpicc MPICompress.c ../include/serialFunctions.c -o ../bin/MPI_compress
	mpicc MPIDecompress.c ../include/serialFunctions.c -o ../bin/MPI_decompress

clean:
	rm -f MPI_compress
	rm -f MPI_decompress
