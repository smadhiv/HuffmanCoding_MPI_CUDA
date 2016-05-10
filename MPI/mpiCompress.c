#include "mpi.h"
#include "header.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include<time.h>

// global variables
struct table h_table[256];
unsigned char bit[255];
unsigned char size = 0;
struct analysis huff[512];
struct analysis *head;

// prototypes
void sort(int i, int node, int arr);
void buildtree(int i, int node, int arr);
void bitvalue(struct analysis *root, unsigned char bit[], unsigned char size);
void printtree(struct analysis *tree);
void gpuCompress(unsigned int nints, unsigned char *h_input, unsigned int *h_offset, struct table *h_table);

main(int argc, char* argv[])
{
	int i, j;
	int deviceNameLength = 20;
	unsigned int *frequency, *compblocklength, *h_offset;
	unsigned int datasize, filelength, cpu_time_used, rank, nprocs, nints, compdatalength;
	unsigned int node = 0, arr = 0;
	unsigned char *data, *compdata;
	unsigned char tgt = 0, tgtlength = 0;
	double start, end;
	
	
	
	FILE *uncompressed;

	// query host info
	unsigned char *name = malloc(deviceNameLength * sizeof(unsigned char));
	gethostname(name, deviceNameLength);
	printf("host: %s\n", name);

	// MPI initialization
	MPI_File srcfile, compfile;
	MPI_Status status;
	MPI_Init( &argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Barrier(MPI_COMM_WORLD);
	
	// read input file size
	if(rank == 0){
		start = MPI_Wtime(); // start timer
		uncompressed = fopen(argv[1], "rb");
		fseeko(uncompressed, 0, SEEK_END);
		filelength = ftello(uncompressed);
		fseeko(uncompressed, 0, SEEK_SET);
		fclose(uncompressed);
	}

	// broadcast size of file to all processes
	MPI_Bcast(&filelength, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// calculate file chunk size
	// note: extra adjustment needed for last process b/c possible uneven chunk
	datasize = filelength/nprocs;
	nints = datasize;
	if(rank == (nprocs - 1))
		nints = filelength - ((nprocs - 1) * nints);

	// MPI file I/O: read
	MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &srcfile);
	MPI_File_seek(srcfile, rank * datasize, MPI_SEEK_SET);
	data = malloc(nints * sizeof(unsigned char));	
	MPI_File_read(srcfile, data, nints, MPI_UNSIGNED_CHAR, &status);

	// calculate frequency of symbols
	frequency = calloc(256, sizeof(unsigned int));
	for (i = 0; i < nints; i++)
		frequency[data[i]]++;

	// initialize nodes and build Huffman tree
	for (i = 0; i < 256; i++)
		if (frequency[i] > 0){
			node++;
			huff[node - 1].count = frequency[i];
			huff[node - 1].letter = i;
			huff[node - 1].left = NULL;
			huff[node - 1].right = NULL;
		}
	for (i = 0; i < node - 1; i++){
		arr = 2 * i;
		sort(i, node, arr);
		buildtree(i, node, arr);
	}
	
	// build dictionary (h_table)
	bitvalue(head, bit, size);

	// calculate byte offsets (h_offset) and round to nearest byte
	h_offset = malloc((nints + 1)*sizeof(unsigned int));
	h_offset[0] = 0;
	for(i = 0; i < nints; i++)
		h_offset[i + 1] = h_table[data[i]].size + h_offset[i];
	if(h_offset[nints] % 8 != 0)
		h_offset[nints] = h_offset[nints] + (8 - (h_offset[nints] % 8));
	
	// launch GPU handler
	// note: this is not the kernel but a C wrapper for kernel
	gpuCompress(nints, data, h_offset, h_table);
	
	// calculate length of compressed data
	compblocklength = malloc(nprocs * sizeof(unsigned int));
	compdatalength = h_offset[nints] / 8 + 1024;
	compblocklength[rank] = compdatalength;

	// send the length of each compressed chunk to process 0
	MPI_Gather(&compdatalength, 1, MPI_UNSIGNED, compblocklength, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// update the data to reflect offsets
	if(rank == 0){
		compblocklength[0] = (nprocs + 2) * 4 + compblocklength[0];
		for(i = 1; i < nprocs; i++)
			compblocklength[i] = compblocklength[i] + compblocklength[i-1];
		for(i = (nprocs - 1); i > 0; i--)
			compblocklength[i] = compblocklength[i - 1];
		compblocklength[0] = (nprocs + 2) * 4;
	}

	// broadcast size of each compressed chunk back to all the processes
	MPI_Bcast(compblocklength, nprocs, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// MPI file I/O: write
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &compfile);
	if(rank == 0){
		MPI_File_write(compfile, &filelength, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
		MPI_File_write(compfile, &nprocs, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
		MPI_File_write(compfile, compblocklength, nprocs, MPI_UNSIGNED, MPI_STATUS_IGNORE);
	}
	MPI_File_seek(compfile, compblocklength[rank], MPI_SEEK_SET);
	MPI_File_write(compfile, frequency, 256, MPI_UNSIGNED, MPI_STATUS_IGNORE);
	MPI_File_write(compfile, data, (compdatalength - 1024), MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
	
	// cleanup and timing
	MPI_File_close(&compfile); 	
	MPI_File_close(&srcfile);
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank==0){
		end = MPI_Wtime(); // end timer
		printf("Total time = %.3fs\n", end - start);
	}
	MPI_Finalize();
}

// sort nodes based on frequency
void sort(int i, int node, int arr){
	int a, b;
	struct analysis temp;
	for (a = arr; a < node - 1 + i; a++)
		for (b = arr; b < node - 1 + i; b++)
			if (huff[b].count > huff[b + 1].count){
				temp = huff[b];
				huff[b] = huff[b + 1];
				huff[b + 1] = temp;
			}
}

// build tree based on sort result
void buildtree(int i, int node, int arr){
	free(head);
	head = (struct analysis *)malloc(sizeof(struct analysis));
	head->count = huff[arr].count + huff[arr + 1].count;
	head->left = &huff[arr];
	head->right = &huff[arr + 1];
	huff[node + i] = *head;
}

// get bit sequence for each char value
void bitvalue(struct analysis *root, unsigned char bit[], unsigned char size){
	if (root->left){
		bit[size] = 0;
		bitvalue(root->left, bit, size + 1);
	}

	if (root->right){
		bit[size] = 1;
		bitvalue(root->right, bit, size + 1);
	}

	if (root->left == NULL && root->right == NULL){
		printf("%u\n", root->letter);
		h_table[root->letter].size = size;
		memcpy(h_table[root->letter].bit, bit, size * sizeof(unsigned char));
	}
}

// function to print the tree
void printtree(struct analysis *tree){
	if (tree->left != NULL || tree->right != NULL){
		printf("im here\n");
		printtree(tree->left);
		printtree(tree->right);
	}
	else
		printf("%d\t%d\n", tree->letter, tree->count);
}
