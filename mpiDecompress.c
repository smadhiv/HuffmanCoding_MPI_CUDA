#include "mpi.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

// table storing compressed bit sequence

struct table
{
	unsigned char bit[100];
	unsigned char size;
}table[256];

unsigned char bit[100], size;

// huffman tree
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

main(int argc, char *argv[])
{
	unsigned int i, k;
	unsigned int node = 0, arr = 0, compfilelength = 0, compsize = 0, *frequency;
	unsigned int bufsize, filelength, cpu_time_used, rank, nprocs, nblocks, nints, lastnints, tgtdatalength = 0, *compblocklength, offset, blocksize;
	unsigned char *buf, *tgtdata, tgt = 0, tgtlength = 0, compressedbyte, *inputcompressed;
	
	FILE *compressed;
	
	unsigned char *name = (unsigned char*)malloc(20*sizeof(unsigned char));
	gethostname(name, 20);
	printf("host: %s\n", name);

	MPI_Init( &argc, &argv);
	MPI_File compfile, tgtfile;
	MPI_Status status;
	
	// get rank and number of processes value

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	// find length of compressed file
	compblocklength = (unsigned int *)malloc((nprocs + 1) * sizeof(unsigned int));

	
	if(rank==0)
	{
		compressed = fopen(argv[1], "rb");
		fseeko(compressed, 0, SEEK_END);
		compfilelength = ftello(compressed);
		fseeko(compressed, 0, SEEK_SET);
		fread(&filelength, sizeof(unsigned int), 1, compressed);
		fread(&nblocks, sizeof(unsigned int), 1, compressed);
		fread(compblocklength, sizeof(unsigned int), nblocks, compressed);
		compblocklength[nblocks] = compfilelength;
		fclose(compressed);
	}
	
	// broadcast block sizes, filelength no. of blocks
	MPI_Bcast(&filelength, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	MPI_Bcast(compblocklength, (nprocs + 1), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	
	//printf("filelength: %d\nnblocks: %d\n %p\n", filelength, nblocks, compblocklength);

	// read the frequency values

	MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &compfile);
	MPI_File_seek(compfile, compblocklength[rank], MPI_SEEK_SET);
	
	// read the header and fill frequency array

	frequency = (unsigned int *)malloc(256 * sizeof(unsigned int));
	MPI_File_read(compfile, frequency, 256, MPI_UNSIGNED, &status);

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
	
	// calculate size of block
	
	compsize = compblocklength[rank+1] - compblocklength[rank] - 1024;
	//printf("\nmy rank %d and compsize %d\n",rank,compsize);
	// allocate required memory and read the file to memory
	
	inputcompressed = (unsigned char *)malloc((compsize)*sizeof(unsigned char));
	tgtdata = (unsigned char *)malloc((filelength / nprocs)*sizeof(unsigned char));
	MPI_File_seek(compfile, compblocklength[rank] + 1024, MPI_SEEK_SET);
	MPI_File_read(compfile, inputcompressed, compsize, MPI_UNSIGNED_CHAR, &status);
	
	// write the data to file

	current = head;

	for (k = 0; k < compsize; k++)
	{
		if(tgtdatalength == filelength / nprocs && rank != (nprocs - 1))
		{
			break;
		}
		else if(tgtdatalength == filelength - (nprocs - 1) * (filelength / nprocs) && rank == (nprocs - 1))
		{
			break;
		}
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
				tgtdata[tgtdatalength] = current->letter;
				tgtdatalength++;
				current = head;
				if(tgtdatalength == filelength / nprocs && rank != (nprocs - 1))
				{
					break;
				}
				else if(tgtdatalength == filelength - (nprocs - 1) * (filelength / nprocs) && rank == (nprocs - 1))
				{
					break;
				}
				}
			}

			else
			{
				current = current->right;
				if (current->right == NULL)
				{
				tgtdata[tgtdatalength] = current->letter;
				tgtdatalength++;
				current = head;
				if(tgtdatalength == filelength / nprocs && rank != (nprocs - 1))
				{
					break;
				}
				else if(tgtdatalength == filelength - (nprocs - 1) * (filelength / nprocs) && rank == (nprocs - 1))
				{
					break;
				}
				}
			}
		}
	}

//	printf("\n%d %d %d %d\n",tgtdatalength, filelength/nprocs, rank, rank * (filelength/nprocs));
	
//printf("\nmy length is %d rank is %d\n",tgtdatalength,rank);
	
MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &tgtfile);

MPI_File_seek(tgtfile, rank * (filelength/nprocs), MPI_SEEK_SET);

MPI_File_write(tgtfile, tgtdata, tgtdatalength, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
/*
if(rank!=(nprocs-1))
{
	MPI_File_write(tgtfile, tgtdata, filelength / nprocs, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
}
else
{
	printf("\n%d %d %d\n",filelength, filelength/nprocs, tgtdatalength);
	MPI_File_write(tgtfile, tgtdata, filelength - (nprocs-1) * (filelength / nprocs), MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
}
*/


//close open files

MPI_File_close(&compfile); 	
MPI_File_close(&tgtfile);
MPI_Finalize();

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

void bitvalue(struct analysis *root, unsigned char bit[], unsigned char size)
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
		memcpy(table[root->letter].bit, bit, size*sizeof(unsigned char));
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
