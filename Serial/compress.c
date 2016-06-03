#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

struct huffmanDictionary{
	unsigned char bitSequence[255];
	unsigned char bitSequenceLength;
}huffmanDictionary[256];

struct huffmanTree{
	unsigned char letter;
	unsigned int count;
	struct huffmanTree *left, *right;
};

struct huffmanTree *head_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512], temp_huffmanTreeNode;

void sortHuffmanTree(int, int, int);
void buildHuffmanTree(int, int, int);
void buildHuffmanDictionary(struct huffmanTree *, unsigned char *, unsigned char);

main(int argc, char **argv){
	clock_t start, end;
	unsigned int cpu_time_used;
	unsigned int i, j;
	unsigned int distinctCharacterCount, combinedHuffmanNodes, frequency[256], inputFileLength, compressedFileLength;
	unsigned char *inputFileData, *compressedData, writeBit = 0, bitsFilled = 0, bitSequence[255], bitSequenceLength = 0;
	FILE *inputFile, *compressedFile;
	
	// start time measure
	start = clock();
	
	// read input file, get filelength and data
	inputFile = fopen(argv[1], "rb");
	fseek(inputFile, 0, SEEK_END);
	inputFileLength = ftell(inputFile);
	fseek(inputFile, 0, SEEK_SET);
	inputFileData = malloc(inputFileLength * sizeof(unsigned char));
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
	
	// write src filelength, header and compressed data to output file
	compressedFile = fopen(argv[2], "wb");
	fwrite(&inputFileLength, sizeof(unsigned int), 1, compressedFile);
	fwrite(frequency, sizeof(unsigned int), 256, compressedFile);
	fwrite(compressedData, sizeof(unsigned char), compressedFileLength, compressedFile);
	fclose(compressedFile);

	// calculate run duration
	end = clock();
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("\ntime taken %d seconds and %d milliseconds\n\n", cpu_time_used / 1000, cpu_time_used % 1000);

}

// sort nodes based on frequency
void sortHuffmanTree(int i, int distinctCharacterCount, int mergedHuffmanNodes){
	int a, b;
	for (a = mergedHuffmanNodes; a < distinctCharacterCount - 1 + i; a++){
		for (b = mergedHuffmanNodes; b < distinctCharacterCount - 1 + i; b++){
			if (huffmanTreeNode[b].count > huffmanTreeNode[b + 1].count){
				temp_huffmanTreeNode = huffmanTreeNode[b];
				huffmanTreeNode[b] = huffmanTreeNode[b + 1];
				huffmanTreeNode[b + 1] = temp_huffmanTreeNode;
			}
		}
	}
}

// build tree based on sort result
void buildHuffmanTree(int i, int distinctCharacterCount, int mergedHuffmanNodes){
	free(head_huffmanTreeNode);
	head_huffmanTreeNode = malloc(sizeof(struct huffmanTree));
	head_huffmanTreeNode->count = huffmanTreeNode[mergedHuffmanNodes].count + huffmanTreeNode[mergedHuffmanNodes + 1].count;
	head_huffmanTreeNode->left = &huffmanTreeNode[mergedHuffmanNodes];
	head_huffmanTreeNode->right = &huffmanTreeNode[mergedHuffmanNodes + 1];
	huffmanTreeNode[distinctCharacterCount + i] = *head_huffmanTreeNode;
}

// get bitSequence sequence for each char value
void buildHuffmanDictionary(struct huffmanTree *root, unsigned char *bitSequence, unsigned char bitSequenceLength){
	if (root->left){
		bitSequence[bitSequenceLength] = 0;
		buildHuffmanDictionary(root->left, bitSequence, bitSequenceLength + 1);
	}

	if (root->right){
		bitSequence[bitSequenceLength] = 1;
		buildHuffmanDictionary(root->right, bitSequence, bitSequenceLength + 1);
	}

	if (root->left == NULL && root->right == NULL){
		huffmanDictionary[root->letter].bitSequenceLength = bitSequenceLength;
		memcpy(huffmanDictionary[root->letter].bitSequence, bitSequence, bitSequenceLength * sizeof(unsigned char));
	}
}

