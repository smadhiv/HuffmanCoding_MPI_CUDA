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
struct huffmanTree *head_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512];

int main(int argc, char *argv[]){
	clock_t start, end;
	unsigned int cpu_time_used;
	unsigned int rank, numProcesses, numCompressedBlocks, compBlockLength;
	unsigned int i, j;
	unsigned int *compBlockLengthArray;
	unsigned int decompBlockLengthCounter, distinctCharacterCount, outputFileLength, combinedHuffmanNodes, frequency[256], compressedFileLength;
	unsigned char currentInputBit, currentInputByte, *compressedData, *outputData, bitSequence[255], bitSequenceLength = 0;
	struct huffmanTree *current_huffmanTreeNode;
	FILE *compressedFile;
	
	MPI_Init( &argc, &argv);
	MPI_File mpi_compressedFile, mpi_outputFile;
	MPI_Status status;
	
	// get rank and number of processes value
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

	// find length of compressedFile file
	compBlockLengthArray = (unsigned int *)malloc((numProcesses + 1) * sizeof(unsigned int));

	
	if(rank == 0){
		compressedFile = fopen(argv[1], "rb");
		fseek(compressedFile, 0, SEEK_END);
		compressedFileLength = ftell(compressedFile);
		fseek(compressedFile, 0, SEEK_SET);
		fread(&outputFileLength, sizeof(unsigned int), 1, compressedFile);
		fread(&numCompressedBlocks, sizeof(unsigned int), 1, compressedFile);
		fread(compBlockLengthArray, sizeof(unsigned int), numCompressedBlocks, compressedFile);
		compBlockLengthArray[numCompressedBlocks] = compressedFileLength;
		fclose(compressedFile);
	}
	
	// broadcast block sizes, outputFileLength no. of blocks
	MPI_Bcast(&outputFileLength, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	MPI_Bcast(compBlockLengthArray, (numProcesses + 1), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// read the frequency values
	MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_compressedFile);
	MPI_File_seek(mpi_compressedFile, compBlockLengthArray[rank], MPI_SEEK_SET);
	
	// read the header and fill frequency array
	MPI_File_read(mpi_compressedFile, frequency, 256, MPI_UNSIGNED, &status);

	// start clock
	if(rank == 0){
		start = clock();
	}
	
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

	// build huffmanDictionary having the bitSequence sequence and its length
	buildHuffmanDictionary(head_huffmanTreeNode, bitSequence, bitSequenceLength);

	
	// calculate size of block
	compBlockLength = compBlockLengthArray[rank + 1] - compBlockLengthArray[rank] - 1024;
	
	compressedData = (unsigned char *)malloc((compBlockLength) * sizeof(unsigned char));
	outputData = (unsigned char *)malloc((outputFileLength / numProcesses) * sizeof(unsigned char));
	
	MPI_File_seek(mpi_compressedFile, compBlockLengthArray[rank] + 1024, MPI_SEEK_SET);
	MPI_File_read(mpi_compressedFile, compressedData, compBlockLength, MPI_UNSIGNED_CHAR, &status);
	
	// write the data to file
	current_huffmanTreeNode = head_huffmanTreeNode;
	for (i = 0; i < compBlockLength; i++){
		if(decompBlockLengthCounter == outputFileLength / numProcesses && rank != (numProcesses - 1)){
			break;
		}
		else if(decompBlockLengthCounter == outputFileLength - (numProcesses - 1) * (outputFileLength / numProcesses) && rank == (numProcesses - 1)){
			break;
		}
		currentInputByte = compressedData[i];
		for (j = 0; j < 8; j++){
			currentInputBit = currentInputByte & 0200;
			currentInputByte = currentInputByte << 1;
			if (currentInputBit == 0){
				current_huffmanTreeNode = current_huffmanTreeNode->left;
				if (current_huffmanTreeNode->left == NULL){
					outputData[decompBlockLengthCounter] = current_huffmanTreeNode->letter;
					decompBlockLengthCounter++;
					current_huffmanTreeNode = head_huffmanTreeNode;
					if(decompBlockLengthCounter == outputFileLength / numProcesses && rank != (numProcesses - 1)){
						break;
					}
					else if(decompBlockLengthCounter == outputFileLength - (numProcesses - 1) * (outputFileLength / numProcesses) && rank == (numProcesses - 1)){
						break;
					}
				}
			}
			else{
				current_huffmanTreeNode = current_huffmanTreeNode->right;
				if (current_huffmanTreeNode->right == NULL){
					outputData[decompBlockLengthCounter] = current_huffmanTreeNode->letter;
					decompBlockLengthCounter++;
					current_huffmanTreeNode = head_huffmanTreeNode;
					if(decompBlockLengthCounter == outputFileLength / numProcesses && rank != (numProcesses - 1)){
						break;
					}
					else if(decompBlockLengthCounter == outputFileLength - (numProcesses - 1) * (outputFileLength / numProcesses) && rank == (numProcesses - 1)){
						break;
					}
				}
			}
		}
	}

	// get time
	if(rank == 0){
		end = clock();
		cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
		printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	}
	
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_outputFile);
	MPI_File_seek(mpi_outputFile, rank * (outputFileLength / numProcesses), MPI_SEEK_SET);
	MPI_File_write(mpi_outputFile, outputData, decompBlockLengthCounter, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
	
	//close open files
	MPI_File_close(&mpi_compressedFile); 	
	MPI_File_close(&mpi_outputFile);
	
	free(compBlockLengthArray);
	free(outputData);
	free(compressedData);	
	MPI_Finalize();
	return 0;
}
