//Sriram Madhivanan
//Struct of Arrays
//Constant memory if dictinary goes beyond 191 bits
//Max possible shared memory
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../Huffman/huffman.h"

__global__ void compress(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, struct huffmanDict *d_huffmanDictionary, 
						 unsigned char *d_byteCompressedData, unsigned int d_inputFileLength, unsigned int constMemoryFlag);

__global__ void compress(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, struct huffmanDict *d_huffmanDictionary, 
						 unsigned char *d_byteCompressedData, unsigned char *d_temp_overflow, unsigned int d_inputFileLength, unsigned int constMemoryFlag, 
						 unsigned int overflowPosition);

#define block_size 1024

int main(int argc, char **argv){
	unsigned int i;
	unsigned int distinctCharacterCount, combinedHuffmanNodes, inputFileLength, frequency[256];
	unsigned char *inputFileData, bitSequenceLength = 0, bitSequence[255];
	unsigned int cpu_time_used;
	FILE *inputFile;
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

	printf("calling wrapper\n");
	wrapperGPU(&argv[2], inputFileData, inputFileLength);

	// calculate run duration
	end = clock();
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("\nTime taken: %d seconds and %d milliseconds\n\n", cpu_time_used / 1000, cpu_time_used % 1000);

	return 0;
}