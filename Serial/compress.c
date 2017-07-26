#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include "../include/serialHeader.h"

struct huffmanDictionary huffmanDictionary[256];
struct huffmanTree *head_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512];

int main(int argc, char **argv){
	clock_t start, end;
	unsigned int cpu_time_used;
	unsigned int i, j;
	unsigned int distinctCharacterCount, combinedHuffmanNodes, frequency[256], inputFileLength, compressedFileLength;
	unsigned char *inputFileData, *compressedData, writeBit = 0, bitsFilled = 0, bitSequence[255], bitSequenceLength = 0;
	FILE *inputFile, *compressedFile;
	
	// read input file, get filelength and data
	inputFile = fopen(argv[1], "rb");
	fseek(inputFile, 0, SEEK_END);
	inputFileLength = ftell(inputFile);
	fseek(inputFile, 0, SEEK_SET);
	inputFileData = malloc(inputFileLength * sizeof(unsigned char));
	fread(inputFileData, sizeof(unsigned char), inputFileLength, inputFile);
	fclose(inputFile);	

	// start time measure
	start = clock();
	
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

	if(distinctCharacterCount == 1){
          head_huffmanTreeNode = &huffmanTreeNode[0];
        }
	
	// build table having the bitSequence sequence and its length
	buildHuffmanDictionary(head_huffmanTreeNode, bitSequence, bitSequenceLength);

	// compress
	compressedData = malloc(inputFileLength * sizeof(unsigned char));
	compressedFileLength = 0;
	for (i = 0; i < inputFileLength; i++){
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
				compressedData[compressedFileLength] = writeBit;
				bitsFilled = 0;
				writeBit = 0;
				compressedFileLength++;
			}
		}
	}

	if (bitsFilled != 0){
		for (i = 0; (unsigned char)i < 8 - bitsFilled; i++){
			writeBit = writeBit << 1;
		}
		compressedData[compressedFileLength] = writeBit;
		compressedFileLength++;
	}

	// calculate run duration
	end = clock();
	
	// write src filelength, header and compressed data to output file
	compressedFile = fopen(argv[2], "wb");
	fwrite(&inputFileLength, sizeof(unsigned int), 1, compressedFile);
	fwrite(frequency, sizeof(unsigned int), 256, compressedFile);
	fwrite(compressedData, sizeof(unsigned char), compressedFileLength, compressedFile);
	fclose(compressedFile);

	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	free(inputFileData);
	free(compressedData);
	return 0;
}



