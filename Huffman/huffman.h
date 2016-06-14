struct huffmanDict
{
	unsigned char bitSequence[256][191];
	unsigned char bitSequenceLength[256];
};
struct huffmanTree
{
	unsigned char letter;
	unsigned int count;
	struct huffmanTree *left, *right;
};

extern struct huffmanTree *head_huffmanTreeNode, *current_huffmanTreeNode;
extern struct huffmanTree huffmanTreeNode[512], temp_huffmanTreeNode;
extern unsigned char bitSequenceConstMemory[256][255];
extern unsigned int constMemoryFlag;
extern struct huffmanDict huffmanDictionary;
extern unsigned int frequency[256];

extern "C" {
	void sortHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes);
	void buildHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes);
	void buildHuffmanDictionary(struct huffmanTree *root, unsigned char *bitSequence, unsigned char bitSequenceLength);
	int wrapperGPU(char **file, unsigned char *inputFileData, int inputFileLength);
}