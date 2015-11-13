// Table has huffman code for each ascii value
struct table
{
	unsigned char bit[60];
	unsigned char size;
}h_table[256];

unsigned char bit[100], size = 0;

// Stores huffman tree
struct analysis
{
	unsigned char letter;
	unsigned int count;
	struct analysis *left, *right;
};
struct analysis *head, *current;
struct analysis huff[512], temp;

// functions for sorting, tree building, storing bit value and printing trree
void sort(int, int, int);
void buildtree(int, int, int);
void bitvalue(struct analysis *, unsigned char bit[], unsigned char);
void printtree(struct analysis *);

// cuda function prototype
void gpuCompress(unsigned int nints, unsigned char *h_input, unsigned int *h_offset, struct table *h_table);
