#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

struct table
{
	unsigned char bit[255];
	unsigned char size;
}table[256];

unsigned int bit[255], size;
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
void bitvalue(struct analysis *, unsigned char bit[], unsigned char);

//void printtree(struct analysis *);


main(int argc, char *argv[]){
	clock_t start, end;
	unsigned int i, j, node = 0, arr = 0, filelength, frequency[256], compressedlength = 0, cpu_time_used, outSize = 0;
	unsigned char *uncompressed, tgt = 0, tgtlength = 0, bit[255], size = 0, *compressedData;
	FILE *source, *compressed;
	
	// start time measure

	start = clock();

	// open source and target compressed file

	compressed = fopen(argv[2], "wb");
	source = fopen(argv[1], "rb");

	// find length of source file

	fseek(source, 0, SEEK_END);
	filelength = ftell(source);
	fseek(source, 0, SEEK_SET);

	// allocate required memory and read the file to memory

	uncompressed = malloc(filelength*sizeof(unsigned char));
	compressedData = malloc(filelength*sizeof(unsigned char));
	fread(uncompressed, sizeof(unsigned char), filelength, source);

	// find the frequency of each symbols

	for (i = 0; i<256; i++){
		frequency[i] = 0;
	}
	for (i = 0; i<filelength; i++){
		frequency[uncompressed[i]]++;
	}

	// initialize nodes of huffman tree

	for (i = 0; i<256; i++){
		if (frequency[i]>0){
			huff[node].count = frequency[i];
			huff[node].letter = i;
			huff[node].left = NULL;
			huff[node].right = NULL;
			node++;
		}
	}

	// build tree 
	for (i = 0; i < node - 1; i++){
		arr = 2 * i;
		sort(i, node, arr);
		buildtree(i, node, arr);
	}
	
	// build table having the bit sequence and its length

	bitvalue(head, bit, size);

	// write the header to the file 
	
	fwrite(&filelength, sizeof(unsigned int), 1, compressed);
	fwrite(frequency, sizeof(unsigned int), 256, compressed);

	// compress

	for (i = 0; i < filelength; i++){
		for (j = 0; j < table[uncompressed[i]].size; j++){
			if (table[uncompressed[i]].bit[j] == 0){
				tgt = tgt << 1;
				tgtlength++;
			}
			else{
				tgt = (tgt << 1) | 01;
				tgtlength++;
			}
			if (tgtlength == 8){
				compressedData[compressedlength] = tgt;
				tgtlength = 0;
				tgt = 00;
				compressedlength++;
			}
		}
	}

	if (tgtlength != 0){
		for (i = 0; (unsigned char)i < 8 - tgtlength; i++){
			tgt = tgt << 1;
		}
		compressedData[compressedlength] = tgt;
		compressedlength++;
	}
	fwrite(compressedData, sizeof(unsigned char), compressedlength, compressed);
	
	// close the file

	fclose(compressed);

	// calculate run duration

	end = clock();
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("\ntime taken %d seconds and %d milliseconds\n\n", cpu_time_used / 1000, cpu_time_used % 1000);

}

// sort nodes based on frequency

void sort(int i, int node, int arr){
	int a, b;
	for (a = arr; a < node - 1 + i; a++){
		for (b = arr; b < node - 1 + i; b++){
			if (huff[b].count > huff[b + 1].count){
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
	head = malloc(sizeof(struct analysis));
	head->count = huff[arr].count + huff[arr + 1].count;
	head->left = &huff[arr];
	head->right = &huff[arr + 1];
	huff[node + i] = *head;
}

// get bit sequence for each char value

void bitvalue(struct analysis *root, unsigned char bit[], unsigned char size)
{
	//int i;
	if (root->left){
		bit[size] = 0;
		bitvalue(root->left, bit, size + 1);
	}

	if (root->right){
		bit[size] = 1;
		bitvalue(root->right, bit, size + 1);
	}

	if (root->left == NULL && root->right == NULL){
		table[root->letter].size = size;
		memcpy(table[root->letter].bit, bit, size * sizeof(unsigned char));
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
