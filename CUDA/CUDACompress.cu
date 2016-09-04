/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//GPU Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include "../include/parallelHeader.h"
#define block_size 1024
#define MAX_GPU_RUNS 10

struct huffmanTree *head_huffmanTreeNode, *current_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512], temp_huffmanTreeNode;
unsigned char bitSequenceConstMemory[256][255];
struct huffmanDictionary huffmanDictionary;
unsigned int constMemoryFlag = 0;

int main(int argc, char **argv){
	unsigned int i;
	unsigned int distinctCharacterCount, combinedHuffmanNodes, inputFileLength, frequency[256];
	unsigned char *inputFileData, bitSequenceLength = 0, bitSequence[255];
	unsigned int *compressedDataOffset, cpu_time_used;
	unsigned int integerOverflowFlag;
	FILE *inputFile, *compressedFile;
	long unsigned int mem_free, mem_total;
	long unsigned int mem_req, mem_offset, mem_data;
	int numKernelRuns;
	clock_t start, end;
	// start time measure
	start = clock();
	
	// read input file, get inputFileLength and data
	inputFile = fopen(argv[1], "rb");
	fseek(inputFile, 0, SEEK_END);
	inputFileLength = ftell(inputFile);
	fseek(inputFile, 0, SEEK_SET);
	inputFileData = (unsigned char *)malloc(inputFileLength * sizeof(unsigned char));
	fread(inputFileData, sizeof(unsigned char), inputFileLength, inputFile);
	fclose(inputFile);
	
	// find the frequency of each symbols
	for (i = 0; i < 256; i++){
		frequency[i] = 0;
	}
	for (i = 0; i < inputFileLength; i++){
		frequency[inputFileData[i]]++;
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
	mem_data = inputFileLength + (inputFileLength + 1) * sizeof(unsigned int) + sizeof(huffmanDictionary);
	
	if(mem_free < mem_data + (mem_offset / MAX_GPU_RUNS)){
		printf("\nExiting : Not enough memory on GPU\nmem_free = %lu\nmin_mem_req = %lu\n", mem_free, (mem_data + (mem_offset / MAX_GPU_RUNS)));
		return -1;
	}
	mem_req = mem_free - mem_data - 10 * 1024 * 1024;
	numKernelRuns = ceil((double)mem_offset / mem_req);
	integerOverflowFlag = mem_req + 255 <= UINT_MAX || mem_offset + 255 <= UINT_MAX ? 0 : 1;
	
	// generate data offset array
	compressedDataOffset = (unsigned int *)malloc((inputFileLength + 1) * sizeof(unsigned int));

	// launch kernel
    lauchCUDAHuffmanCompress(inputFileData, compressedDataOffset, inputFileLength, numKernelRuns, integerOverflowFlag, mem_req);

	// write src inputFileLength, header and compressed data to output file
	compressedFile = fopen(argv[2], "wb");
	fwrite(&inputFileLength, sizeof(unsigned int), 1, compressedFile);
	fwrite(frequency, sizeof(unsigned int), 256, compressedFile);
	fwrite(inputFileData, sizeof(unsigned char), mem_offset / 8, compressedFile);
	fclose(compressedFile);	
	
	// calculate run duration
	end = clock();
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	return 0;
}