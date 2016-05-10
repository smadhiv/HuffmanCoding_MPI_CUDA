#include "mpi.h"
#include "header_mpi.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include<time.h>

main(int argc, char* argv[])
{
	int i, j, deviceNameLength = 20;
	unsigned int *frequency, *compblocklength, *h_offset;
	unsigned int datasize, filelength, cpu_time_used, rank, nprocs, nints, compdatalength, node = 0, arr = 0;
	unsigned char *data, *compdata;
	unsigned char tgt = 0, tgtlength = 0;
	FILE *uncompressed;
	double start, end;

	unsigned char *name = malloc(deviceNameLength * sizeof(unsigned char));
	gethostname(name, deviceNameLength);
	printf("host: %s\n", name);

	MPI_Init( &argc, &argv);
	MPI_File srcfile, compfile;
	MPI_Status status;

	// get rank and number of processes value
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Barrier(MPI_COMM_WORLD);
	
	// get file size
	if(rank == 0){
		start = MPI_Wtime();
		
		uncompressed = fopen(argv[1], "rb");
		fseeko(uncompressed, 0, SEEK_END);
		filelength = ftello(uncompressed);
		fseeko(uncompressed, 0, SEEK_SET);
		fclose(uncompressed);
	}

	// broadcast size of file to all the processes
	MPI_Bcast(&filelength, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// get file chunk size
	datasize = filelength/nprocs;
	nints = datasize;
	if(rank == (nprocs - 1)){
		nints = filelength - ((nprocs - 1) * nints);
	}

	// find the frequency of each symbols
	frequency = calloc(256, sizeof(unsigned int));
	// open file in each process and read data and allocate memory for compressed data
	MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &srcfile);
	MPI_File_seek(srcfile, rank * datasize, MPI_SEEK_SET);
	data = malloc(nints * sizeof(unsigned char));	
	MPI_File_read(srcfile, data, nints, MPI_UNSIGNED_CHAR, &status);

	for (i = 0; i < nints; i++){
		frequency[data[i]]++;
	}

	compblocklength = malloc(nprocs * sizeof(unsigned int));

	// initialize nodes of huffman tree
	for (i = 0; i < 256; i++){
		if (frequency[i] > 0){
			node++;
			huff[node - 1].count = frequency[i];
			huff[node - 1].letter = i;
			huff[node - 1].left = NULL;
			huff[node - 1].right = NULL;
		}
	}

	// build tree 
	for (i = 0; i < node - 1; i++){
		arr = 2 * i;
		sort(i, node, arr);
		buildtree(i, node, arr);
	}
	
	// build h_table having the bit sequence and its length
	bitvalue(head, bit, size);

	// calculate h_offset
	// this block converts each character to its code book value in bytes and calculate offset
	h_offset = malloc((nints + 1)*sizeof(unsigned int));
	h_offset[0] = 0;
	for(i = 0; i < nints; i++){
		h_offset[i + 1] = h_table[data[i]].size + h_offset[i];
	}
	
	if(h_offset[nints] % 8 != 0){
		h_offset[nints] = h_offset[nints] + (8 - (h_offset[nints] % 8));
	}
	
	// launch GPU kernel
	gpuCompress(nints, data, h_offset, h_table);
	
	// calculate length of compressed data
	compdatalength = h_offset[nints] / 8 + 1024;
	compblocklength[rank] = compdatalength;

	// send the length of each process to process 0
	MPI_Gather(&compdatalength, 1, MPI_UNSIGNED, compblocklength, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// update the data to reflect the offset
	if(rank == 0){
		compblocklength[0] = (nprocs + 2) * 4 + compblocklength[0];
		for(i = 1; i < nprocs; i++){
			compblocklength[i] = compblocklength[i] + compblocklength[i-1];
		}
		for(i = (nprocs - 1); i > 0; i--){
			compblocklength[i] = compblocklength[i - 1];
		}
		compblocklength[0] = (nprocs + 2) * 4;
	}

	// broadcast size of each compressed data block to all the processes 
	MPI_Bcast(compblocklength, nprocs, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	// write data to file
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &compfile);
	if(rank == 0){
		MPI_File_write(compfile, &filelength, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
		MPI_File_write(compfile, &nprocs, 1, MPI_UNSIGNED, MPI_STATUS_IGNORE);
		MPI_File_write(compfile, compblocklength, nprocs, MPI_UNSIGNED, MPI_STATUS_IGNORE);
	}
	MPI_File_seek(compfile, compblocklength[rank], MPI_SEEK_SET);
	MPI_File_write(compfile, frequency, 256, MPI_UNSIGNED, MPI_STATUS_IGNORE);
	MPI_File_write(compfile, data, (compdatalength - 1024), MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
	
	// close open files
	MPI_File_close(&compfile); 	
	MPI_File_close(&srcfile);
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank==0){
		end = MPI_Wtime();
		printf("Total time = %.3fs\n", end - start);
	}
	MPI_Finalize();
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
	else{
		printf("%d\t%d\n", tree->letter, tree->count);
	}
}
