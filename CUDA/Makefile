all: main

main: finalGPU.o compress.o huffman.o
	nvcc -o main finalGPU.o -L /opt/cuda-toolkit/7.0.28/lib64 -lcudart compress.o huffman.o 

finalGPU.o: ../Huffman/huffman.h
	nvcc -c finalGPU.cu

compress.o: ../Huffman/huffman.h
	nvcc -c compress.cu -arch=sm_35

huffman.o: ../Huffman/huffman.h
	g++ -c ../Huffman/huffman.c

clean:
	rm -rf *.o