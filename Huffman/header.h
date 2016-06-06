struct huffmanDictionary
{
	unsigned char bitSequence[256][191];
	unsigned char bitSequenceLength[256];
}huffmanDictionary;

struct huffmanTree
{
	unsigned char letter;
	unsigned int count;
	struct huffmanTree *left, *right;
};