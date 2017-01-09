/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//Functions used by serial and MPI-only implementations
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "serialHeader.h"

// sort nodes based on frequency
void sortHuffmanTree(int i, int distinctCharacterCount, int mergedHuffmanNodes){
	int a, b;
	for (a = mergedHuffmanNodes; a < distinctCharacterCount - 1 + i; a++){
		for (b = mergedHuffmanNodes; b < distinctCharacterCount - 1 + i; b++){
			if (huffmanTreeNode[b].count > huffmanTreeNode[b + 1].count){
				struct huffmanTree temp_huffmanTreeNode = huffmanTreeNode[b];
				huffmanTreeNode[b] = huffmanTreeNode[b + 1];
				huffmanTreeNode[b + 1] = temp_huffmanTreeNode;
			}
		}
	}
}

// build tree based on sort result
void buildHuffmanTree(int i, int distinctCharacterCount, int mergedHuffmanNodes){
	huffmanTreeNode[distinctCharacterCount + i].count = huffmanTreeNode[mergedHuffmanNodes].count + huffmanTreeNode[mergedHuffmanNodes + 1].count;
	huffmanTreeNode[distinctCharacterCount + i].left = &huffmanTreeNode[mergedHuffmanNodes];
	huffmanTreeNode[distinctCharacterCount + i].right = &huffmanTreeNode[mergedHuffmanNodes + 1];
	head_huffmanTreeNode = &(huffmanTreeNode[distinctCharacterCount + i]);
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