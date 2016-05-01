#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

struct table
{
	unsigned int bit[100];
	unsigned int size;
}table[256];

unsigned int bit[100], size;
struct analysis
{
	unsigned char letter;
	unsigned int count;
	struct analysis *left, *right;
};

struct analysis *head, *current;
struct analysis huff[512], temp;

void sort(int, int, int);
void buildtree(int, int, int);
void printtree(struct analysis *);
void bitvalue(struct analysis *, unsigned int bit[], int);


main(int argc, char *argv[])
{
	clock_t start, end;
	unsigned int i, k, node = 0, outchar = 0, filelength, arr = 0, bit[100], size = 0, *frequency, compressedlength = 0, cpu_time_used;
	unsigned char tgt = 0, tgtlength = 0, compressedbyte, *inputcompressed;
	FILE *compressed, *output;

	// start time measure

	start = clock();

	// open source compressed file and target file

	compressed = fopen(argv[1], "rb");
	output = fopen(argv[2], "wb");

	// find length of compressed file

	fseek(compressed, 0, SEEK_END);
	compressedlength = ftell(compressed);
	fseek(compressed, 0, SEEK_SET);

	// read the header and fill frequency array

	frequency = (unsigned int *)malloc(256 * sizeof(unsigned int));
	fread(&filelength, sizeof(unsigned int), 1, compressed);
	printf("\n%d",filelength);
	fread(frequency, 256 * sizeof(unsigned int), 1, compressed);

	// initialize nodes of huffman tree

	for (i = 0; i<256; i++)
	{
		if (frequency[i]>0)
		{
			node++;
			huff[node - 1].count = frequency[i];
			huff[node - 1].letter = i;
			huff[node - 1].left = NULL;
			huff[node - 1].right = NULL;
		}
	}

	// build tree 

	for (i = 0; i < node - 1; i++)
	{
		arr = 2 * i;
		sort(i, node, arr);
		buildtree(i, node, arr);
	}

	// build table having the bit sequence and its length

	bitvalue(head, bit, size);

	// forward to the position after header

	fseek(compressed, 1028, SEEK_SET);

	// get the compressed data length (minus header)

	compressedlength = compressedlength - 1028;

	// allocate required memory and read the file to memory

	inputcompressed = (unsigned char *)malloc((compressedlength)*sizeof(unsigned char));
	fread(inputcompressed, sizeof(unsigned char), (compressedlength), compressed);

	// write the data to file

	current = head;

	for (k = 0; k < compressedlength; k++)
	{
		compressedbyte = inputcompressed[k];
		for (i = 0; i<8; i++)
		{
			tgt = compressedbyte & 0200;
			compressedbyte = compressedbyte << 1;
			if (tgt == 0)
			{
				current = current->left;
				if (current->left == NULL)
				{
					fputc(current->letter, output);
					current = head;
					outchar++;
				}
			}

			else
			{
				current = current->right;
				if (current->right == NULL)
				{
					fputc(current->letter, output);
					current = head;
					outchar++;
				}

			}
			if(outchar == filelength)
			{
			break;
			}
		}
	}

	// close the file

	fclose(output);

	//display runtime

	end = clock();
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("\ntime taken %d seconds and %d milliseconds\n\n", cpu_time_used / 1000, cpu_time_used % 1000);
}

// sort nodes based on frequency

void sort(int i, int node, int arr)
{
	int a, b;
	for (a = arr; a < node - 1 + i; a++)
	{
		for (b = arr; b < node - 1 + i; b++)
		{
			if (huff[b].count > huff[b + 1].count)
			{
				temp = huff[b];
				huff[b] = huff[b + 1];
				huff[b + 1] = temp;
			}
		}
	}
}

// build tree based on sort result

void buildtree(int i, int node, int arr)
{
	free(head);
	head = (struct analysis *)malloc(sizeof(struct analysis));
	head->count = huff[arr].count + huff[arr + 1].count;
	head->left = &huff[arr];
	head->right = &huff[arr + 1];
	huff[node + i] = *head;
}

// get bit sequence for each char value

void bitvalue(struct analysis *root, unsigned int bit[], int size)
{
	//int i;
	if (root->left)
	{
		bit[size] = 0;
		bitvalue(root->left, bit, size + 1);
	}

	if (root->right)
	{
		bit[size] = 1;
		bitvalue(root->right, bit, size + 1);
	}

	if (root->left == NULL && root->right == NULL)
	{
		table[root->letter].size = size;
		memcpy(table[root->letter].bit, bit, size*sizeof(unsigned int));
		/*
		printf("\nchar %c\t size %d\n", root->letter, size);
		for (i = 0; i < size;i++)
		{
		printf("%d", table[root->letter].bit[i]);
		}
		*/
	}
}

// function to print the tree

/*
void printtree(struct analysis *tree)
{
if (tree->left != NULL || tree->right != NULL)
{
printtree(tree->left);
printtree(tree->right);
}
else
{
printf("%d\t%d\n", tree->letter, tree->count);
}
}
*/
