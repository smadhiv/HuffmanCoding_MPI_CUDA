// table struct
struct table
{
	unsigned char bit[255];
	unsigned char size;
};

// tree node struct
struct analysis
{
	unsigned char letter;
	unsigned int count;
	struct analysis *left, *right;
};
