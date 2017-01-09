/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//Functions used for GPU and CUDA-MPI implementations
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "parallelHeader.h"

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// sortHuffmanTree nodes based on frequency
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void sortHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes){
	int a, b;
	for (a = combinedHuffmanNodes; a < distinctCharacterCount - 1 + i; a++){
		for (b = combinedHuffmanNodes; b < distinctCharacterCount - 1 + i; b++){
			if (huffmanTreeNode[b].count > huffmanTreeNode[b + 1].count){
				struct huffmanTree temp_huffmanTreeNode = huffmanTreeNode[b];
				huffmanTreeNode[b] = huffmanTreeNode[b + 1];
				huffmanTreeNode[b + 1] = temp_huffmanTreeNode;
			}
		}
	}
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// build tree based on sortHuffmanTree result
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void buildHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes){
	huffmanTreeNode[distinctCharacterCount + i].count = huffmanTreeNode[combinedHuffmanNodes].count + huffmanTreeNode[combinedHuffmanNodes + 1].count;
	huffmanTreeNode[distinctCharacterCount + i].left = &huffmanTreeNode[combinedHuffmanNodes];
	huffmanTreeNode[distinctCharacterCount + i].right = &huffmanTreeNode[combinedHuffmanNodes + 1];
	head_huffmanTreeNode = &(huffmanTreeNode[distinctCharacterCount + i]);
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// get bitSequence sequence for each char value
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
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
		huffmanDictionary.bitSequenceLength[root->letter] = bitSequenceLength;
		if(bitSequenceLength < 192){
			memcpy(huffmanDictionary.bitSequence[root->letter], bitSequence, bitSequenceLength * sizeof(unsigned char));
		}
		else{
			memcpy(bitSequenceConstMemory[root->letter], bitSequence, bitSequenceLength * sizeof(unsigned char));
			memcpy(huffmanDictionary.bitSequence[root->letter], bitSequence, 191);
			constMemoryFlag = 1;
		}
	}
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// generate data offset array 
// case - single run, no overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength){
	int i;
	compressedDataOffset[0] = 0;
	for(i = 0; i < inputFileLength; i++){
		compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]] + compressedDataOffset[i];
	}
	if(compressedDataOffset[inputFileLength] % 8 != 0){
		compressedDataOffset[inputFileLength] = compressedDataOffset[inputFileLength] + (8 - (compressedDataOffset[inputFileLength] % 8));
	}		
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// generate data offset array 
// case - single run, with overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, unsigned int *integerOverflowIndex, unsigned int *bitPaddingFlag, int numBytes){
	int i, j;
	// calculate compressed data offset - (1048576 is a safe number that will ensure there is no integer overflow in GPU, it should be minimum 8 * number of threads)
	j = 0;
	compressedDataOffset[0] = 0;
	for(i = 0; i < inputFileLength; i++){
		compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]] + compressedDataOffset[i];
		if(compressedDataOffset[i + 1] + numBytes < compressedDataOffset[i]){
			integerOverflowIndex[j] = i;
			if(compressedDataOffset[i] % 8 != 0){
				bitPaddingFlag[j] = 1;
				compressedDataOffset[i + 1] = (compressedDataOffset[i] % 8) + huffmanDictionary.bitSequenceLength[inputFileData[i]];
				compressedDataOffset[i] = compressedDataOffset[i] + (8 - (compressedDataOffset[i] % 8));
			}
			else{
				compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]];			
			}
			j++;
		}
	}
	if(compressedDataOffset[inputFileLength] % 8 != 0){
		compressedDataOffset[inputFileLength] = compressedDataOffset[inputFileLength] + (8 - (compressedDataOffset[inputFileLength] % 8));
	}
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// generate data offset array 
// case - multiple run, no overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, unsigned int *gpuMemoryOverflowIndex, unsigned int *gpuBitPaddingFlag, long unsigned int mem_req){
	int i, j;
	j = 0;
	gpuMemoryOverflowIndex[0] = 0;
	gpuBitPaddingFlag[0] = 0;
	compressedDataOffset[0] = 0;
	for(i = 0; i < inputFileLength; i++){
		compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]] + compressedDataOffset[i];
		if(compressedDataOffset[i + 1] > mem_req){
			gpuMemoryOverflowIndex[j * 2 + 1] = i;
			gpuMemoryOverflowIndex[j * 2 + 2] = i + 1;
			if(compressedDataOffset[i] % 8 != 0){
				gpuBitPaddingFlag[j + 1] = 1;
				compressedDataOffset[i + 1] = (compressedDataOffset[i] % 8) + huffmanDictionary.bitSequenceLength[inputFileData[i]];
				compressedDataOffset[i] = compressedDataOffset[i] + (8 - (compressedDataOffset[i] % 8));
			}
			else{
				compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]];			
			}
			j++;
		}
	}
	if(compressedDataOffset[inputFileLength] % 8 != 0){
		compressedDataOffset[inputFileLength] = compressedDataOffset[inputFileLength] + (8 - (compressedDataOffset[inputFileLength] % 8));
	}
	gpuMemoryOverflowIndex[j * 2 + 1] = inputFileLength;
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// generate data offset array
// case - multiple run, with overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void createDataOffsetArray(unsigned int *compressedDataOffset, unsigned char* inputFileData, unsigned int inputFileLength, unsigned int *integerOverflowIndex, unsigned int *bitPaddingFlag, unsigned int *gpuMemoryOverflowIndex, unsigned int *gpuBitPaddingFlag, int numBytes, long unsigned int mem_req){
	int i, j, k;
	j = 0;
	k = 0;
	compressedDataOffset[0] = 0;
	for(i = 0; i < inputFileLength; i++){
		compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]] + compressedDataOffset[i];
		if(j != 0 && ((long unsigned int)compressedDataOffset[i + 1] + compressedDataOffset[integerOverflowIndex[j - 1]] > mem_req)){
			gpuMemoryOverflowIndex[k * 2 + 1] = i;
			gpuMemoryOverflowIndex[k * 2 + 2] = i + 1;
			if(compressedDataOffset[i] % 8 != 0){
				gpuBitPaddingFlag[k + 1] = 1;
				compressedDataOffset[i + 1] = (compressedDataOffset[i] % 8) + huffmanDictionary.bitSequenceLength[inputFileData[i]];
				compressedDataOffset[i] = compressedDataOffset[i] + (8 - (compressedDataOffset[i] % 8));
			}
			else{
				compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]];			
			}
			k++;
		}
		else if(compressedDataOffset[i + 1] + numBytes < compressedDataOffset[i]){
			integerOverflowIndex[j] = i;
			if(compressedDataOffset[i] % 8 != 0){
				bitPaddingFlag[j] = 1;
				compressedDataOffset[i + 1] = (compressedDataOffset[i] % 8) + huffmanDictionary.bitSequenceLength[inputFileData[i]];
				compressedDataOffset[i] = compressedDataOffset[i] + (8 - (compressedDataOffset[i] % 8));
			}
			else{
				compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]];	
			}
			j++;
		}
	}
	if(compressedDataOffset[inputFileLength] % 8 != 0){
		compressedDataOffset[inputFileLength] = compressedDataOffset[inputFileLength] + (8 - (compressedDataOffset[inputFileLength] % 8));
	}
	gpuMemoryOverflowIndex[j * 2 + 1] = inputFileLength;
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*---------------------------------------------------------------------------------------------------------------------------------------------*/