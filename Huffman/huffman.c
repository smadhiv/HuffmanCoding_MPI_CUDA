#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "huffman.h"

struct huffmanTree *head_huffmanTreeNode, *current_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512], temp_huffmanTreeNode;
unsigned char bitSequenceConstMemory[256][255];
unsigned int constMemoryFlag = 0;
struct huffmanDict huffmanDictionary;

// sortHuffmanTree nodes based on frequency
extern "C" void sortHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes){
	int a, b;
	for (a = combinedHuffmanNodes; a < distinctCharacterCount - 1 + i; a++){
		for (b = combinedHuffmanNodes; b < distinctCharacterCount - 1 + i; b++){
			if (huffmanTreeNode[b].count > huffmanTreeNode[b + 1].count){
				temp_huffmanTreeNode = huffmanTreeNode[b];
				huffmanTreeNode[b] = huffmanTreeNode[b + 1];
				huffmanTreeNode[b + 1] = temp_huffmanTreeNode;
			}
		}
	}
}

// build tree based on sortHuffmanTree result
extern "C" void buildHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes){
	free(head_huffmanTreeNode);
	head_huffmanTreeNode = (struct huffmanTree *)malloc(sizeof(struct huffmanTree));
	head_huffmanTreeNode->count = huffmanTreeNode[combinedHuffmanNodes].count + huffmanTreeNode[combinedHuffmanNodes + 1].count;
	head_huffmanTreeNode->left = &huffmanTreeNode[combinedHuffmanNodes];
	head_huffmanTreeNode->right = &huffmanTreeNode[combinedHuffmanNodes + 1];
	huffmanTreeNode[distinctCharacterCount + i] = *head_huffmanTreeNode;
}

// get bitSequence sequence for each char value
extern "C" void buildHuffmanDictionary(struct huffmanTree *root, unsigned char *bitSequence, unsigned char bitSequenceLength){
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