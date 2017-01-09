/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//MPI Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

#include "mpi.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include "../include/serialHeader.h"

struct huffmanDictionary huffmanDictionary[256];
struct huffmanTree *head_huffmanTreeNode = NULL;
struct huffmanTree huffmanTreeNode[512];

int main(int argc, char* argv[]){
	clock_t start, end;
	unsigned int cpu_time_used;
	unsigned int i, j, rank, numProcesses, blockLength;
	unsigned int *compBlockLengthArray;
	unsigned int distinctCharacterCount, combinedHuffmanNodes, frequency[256], inputFileLength, compBlockLength;
	unsigned char *inputFileData, *compressedData, writeBit = 0, bitsFilled = 0, bitSequence[255], bitSequenceLength = 0;
	FILE *inputFile;

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
	
	compressedData = (unsigned char *)malloc(blockLength * sizeof(unsigned char));	
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

	// compress
	compBlockLength = 0;
	for (i = 0; i < blockLength; i++){
		for (j = 0; j < huffmanDictionary[inputFileData[i]].bitSequenceLength; j++){
			if (huffmanDictionary[inputFileData[i]].bitSequence[j] == 0){
				writeBit = writeBit << 1;
				bitsFilled++;
			}
			else{
				writeBit = (writeBit << 1) | 01;
				bitsFilled++;
			}
			if (bitsFilled == 8){
				compressedData[compBlockLength] = writeBit;
				bitsFilled = 0;
				writeBit = 0;
				compBlockLength++;
			}
		}
	}

	if (bitsFilled != 0){
		for (i = 0; (unsigned char)i < 8 - bitsFilled; i++){
			writeBit = writeBit << 1;
		}
		compressedData[compBlockLength] = writeBit;
		compBlockLength++;
	}

	// calculate length of compressed data
	compBlockLength = compBlockLength + 1024;
	compBlockLengthArray[rank] = compBlockLength;

	// send the length of each process to process 0
	MPI_Gather(&compBlockLength, 1, MPI_UNSIGNED, compBlockLengthArray, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// update the data to reflect the offset
	if(rank == 0){
		compBlockLengthArray[0] = (numProcesses + 2) * 4 + compBlockLengthArray[0];
		for(i = 1; i < numProcesses; i++){
			compBlockLengthArray[i] = compBlockLengthArray[i] + compBlockLengthArray[i - 1];
		}
		for(i = (numProcesses - 1); i > 0; i--){
			compBlockLengthArray[i] = compBlockLengthArray[i - 1];
		}
		compBlockLengthArray[0] = (numProcesses + 2) * 4;
	}

	// broadcast size of each compressed data block to all the processes 
	MPI_Bcast(compBlockLengthArray, numProcesses, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// get time
	if(rank == 0){
		end = clock();
		cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
		printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	}
	
	// write data to file
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_compressedFile);

	if(rank == 0){
		MPI_File_write(mpi_compressedFile, &inputFileLength, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
		MPI_File_write(mpi_compressedFile, &numProcesses, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
		MPI_File_write(mpi_compressedFile, compBlockLengthArray, numProcesses, MPI_UNSIGNED, MPI_STATUS_IGNORE);
	}
	MPI_File_seek(mpi_compressedFile, compBlockLengthArray[rank], MPI_SEEK_SET);
	MPI_File_write(mpi_compressedFile, frequency, 256, MPI_UNSIGNED, MPI_STATUS_IGNORE);
	MPI_File_write(mpi_compressedFile, compressedData, (compBlockLength - 1024), MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);

	// close open files
	MPI_File_close(&mpi_compressedFile); 	
	MPI_File_close(&mpi_inputFile);
	MPI_Barrier(MPI_COMM_WORLD);
	
	free(head_huffmanTreeNode);
	free(compBlockLengthArray);
	free(inputFileData);
	free(compressedData);
	MPI_Finalize();
	return 0;
}



