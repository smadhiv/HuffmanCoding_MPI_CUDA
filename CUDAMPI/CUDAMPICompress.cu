/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//CUDA-MPI Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

#include "mpi.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <limits.h>
#include<time.h>
#include<math.h>
#include "../include/parallelHeader.h"
#define block_size 1024
#define MIN_SCRATCH_SIZE 50 * 1024 * 1024

struct huffmanTree *head_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512];
unsigned char bitSequenceConstMemory[256][255];
struct huffmanDictionary huffmanDictionary;
unsigned int constMemoryFlag = 0;

main(int argc, char* argv[]){
	clock_t start, end;
	int rank, numProcesses;
	unsigned int cpu_time_used;
	unsigned int i, blockLength;
	unsigned int *compressedDataOffset, *compBlockLengthArray;
	unsigned int distinctCharacterCount, combinedHuffmanNodes, inputFileLength, compBlockLength, frequency[256];
	unsigned char *inputFileData, bitSequence[255], bitSequenceLength = 0;
	FILE *inputFile;
	unsigned int integerOverflowFlag;
	long unsigned int mem_free, mem_total;
	long unsigned int mem_req, mem_offset, mem_data;
	int numKernelRuns;
	
	MPI_Init( &argc, &argv);
	MPI_File mpi_inputFile, mpi_compressedFile;
	MPI_Status status;

	// get rank and number of processes value
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

	// get file size
	if(rank == 0){
		inputFile = fopen(argv[1], "rb");
		fseek(inputFile, 0, SEEK_END);
		inputFileLength = ftell(inputFile);
		fseek(inputFile, 0, SEEK_SET);
		fclose(inputFile);
	}

	//broadcast size of file to all the processes 
	MPI_Bcast(&inputFileLength, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// get file chunk size

	blockLength = inputFileLength / numProcesses;

	if(rank == (numProcesses-1)){
		blockLength = inputFileLength - ((numProcesses-1) * blockLength);	
	}
	
	// open file in each process and read data and allocate memory for compressed data
	MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_inputFile);
	MPI_File_seek(mpi_inputFile, rank * blockLength, MPI_SEEK_SET);

	inputFileData = (unsigned char *)malloc(blockLength * sizeof(unsigned char));	
	MPI_File_read(mpi_inputFile, inputFileData, blockLength, MPI_UNSIGNED_CHAR, &status);

	// start clock
	if(rank == 0){
		start = clock();
	}
	
	// find the frequency of each symbols
	for (i = 0; i < 256; i++){
		frequency[i] = 0;
	}
	for (i = 0; i < blockLength; i++){
		frequency[inputFileData[i]]++;
	}
	
	compBlockLengthArray = (unsigned int *)malloc(numProcesses * sizeof(unsigned int));
	
	// initialize nodes of huffman tree
	distinctCharacterCount = 0;
	for (i = 0; i < 256; i++){
		if (frequency[i] > 0){
			huffmanTreeNode[distinctCharacterCount].count = frequency[i];
			huffmanTreeNode[distinctCharacterCount].letter = i;
			huffmanTreeNode[distinctCharacterCount].left = NULL;
			huffmanTreeNode[distinctCharacterCount].right = NULL;
			distinctCharacterCount++;
		}
	}

	// build tree 
	for (i = 0; i < distinctCharacterCount - 1; i++){
		combinedHuffmanNodes = 2 * i;
		sortHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
		buildHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
	}
	
	// build table having the bitSequence sequence and its length
	buildHuffmanDictionary(head_huffmanTreeNode, bitSequence, bitSequenceLength);

	// calculate memory requirements
	// GPU memory
	cudaMemGetInfo(&mem_free, &mem_total);
	
	// offset array requirements
	mem_offset = 0;
	for(i = 0; i < 256; i++){
		mem_offset += frequency[i] * huffmanDictionary.bitSequenceLength[i];
	}
	mem_offset = mem_offset % 8 == 0 ? mem_offset : mem_offset + 8 - mem_offset % 8;
	
	// other memory requirements
	mem_data = blockLength + (blockLength + 1) * sizeof(unsigned int) + sizeof(huffmanDictionary);
	
	if(mem_free - mem_data < MIN_SCRATCH_SIZE){
		printf("\nExiting : Not enough memory on GPU\nmem_free = %lu\nmin_mem_req = %lu\n", mem_free, mem_data + MIN_SCRATCH_SIZE);
		return -1;
	}
	mem_req = mem_free - mem_data - 10 * 1024 * 1024;
	numKernelRuns = ceil((double)mem_offset / mem_req);
	integerOverflowFlag = mem_req + 255 <= UINT_MAX || mem_offset + 255 <= UINT_MAX ? 0 : 1;
	
	// generate data offset array
	compressedDataOffset = (unsigned int *)malloc((blockLength + 1) * sizeof(unsigned int));
	
	// launch kernel
    lauchCUDAHuffmanCompress(inputFileData, compressedDataOffset, blockLength, numKernelRuns, integerOverflowFlag, mem_req);

	// calculate length of compressed data
	compBlockLengthArray = (unsigned int *)malloc(numProcesses * sizeof(unsigned int));
	compBlockLength = mem_offset / 8 + 1024;
	compBlockLengthArray[rank] = compBlockLength;

	// send the length of each compressed chunk to process 0
	MPI_Gather(&compBlockLength, 1, MPI_UNSIGNED, compBlockLengthArray, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// update the data to reflect offsets
	if(rank == 0){
		compBlockLengthArray[0] = (numProcesses + 2) * 4 + compBlockLengthArray[0];
		for(i = 1; i < numProcesses; i++)
			compBlockLengthArray[i] = compBlockLengthArray[i] + compBlockLengthArray[i-1];
		for(i = (numProcesses - 1); i > 0; i--)
			compBlockLengthArray[i] = compBlockLengthArray[i - 1];
		compBlockLengthArray[0] = (numProcesses + 2) * 4;
	}

	// broadcast size of each compressed chunk back to all the processes
	MPI_Bcast(compBlockLengthArray, numProcesses, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// get time
	if(rank == 0){
		end = clock();
		cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
		printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	}
	
	// MPI file I/O: write
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_compressedFile);
	if(rank == 0){
		MPI_File_write(mpi_compressedFile, &inputFileLength, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
		MPI_File_write(mpi_compressedFile, &numProcesses, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
		MPI_File_write(mpi_compressedFile, compBlockLengthArray, numProcesses, MPI_UNSIGNED, MPI_STATUS_IGNORE);
	}
	MPI_File_seek(mpi_compressedFile, compBlockLengthArray[rank], MPI_SEEK_SET);
	MPI_File_write(mpi_compressedFile, frequency, 256, MPI_UNSIGNED, MPI_STATUS_IGNORE);
	MPI_File_write(mpi_compressedFile, inputFileData, (compBlockLength - 1024), MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
	
	//close open files
	MPI_File_close(&mpi_compressedFile); 	
	MPI_File_close(&mpi_inputFile);
	
	free(inputFileData);
	free(compressedDataOffset);
	free(compBlockLengthArray);
	MPI_Finalize();
}
