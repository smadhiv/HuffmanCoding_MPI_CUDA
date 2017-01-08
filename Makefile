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
	make clean -C 'CUDAMPI'
	make clean -C 'CUDA'
	make clean -C 'MPI'
	make clean -C 'Serial'

