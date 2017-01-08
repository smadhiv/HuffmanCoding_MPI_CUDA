all:
	mpicc MPICompress.c ../include/serialFunctions.c -o ../bin/MPI_compress
	mpicc MPIDecompress.c ../include/serialFunctions.c -o ../bin/MPI_decompress

clean:
	if [ -a ../bin/MPI_compress ]; then rm -f ../bin/MPI_compress; fi;
	if [ -a ../bin/MPI_decompress ]; then rm -f ../bin/MPI_decompress; fi;
