all:
	make -C 'CUDAMPI'
	make -C 'CUDA'
	make -C 'MPI'
	make -C 'Serial'
CUDAMPI:
	make -C 'CUDAMPI'
CUDA:
	make -C 'CUDA'
MPI:
	make -C 'MPI'
Serial:
	make -C 'Serial'
clean:
	rm -r bin/

