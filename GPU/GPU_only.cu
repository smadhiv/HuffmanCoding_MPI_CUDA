//Sriram Madhivanan
//Struct of Arrays
//Constant memory if dictinary goes beyond 191 bits
//Max possible shared memory
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

// To handle worst case huffman tree
__constant__ unsigned char d_bitDict[256][255];
unsigned char max_bitDict[256][255];
unsigned int flag = 0;

// struct to store dictionary
struct table_struct
{
	unsigned char bitDict[256][191];
	unsigned char sizeDict[256];
}h_table;

// huffman tree
struct analysis
{
	unsigned char letter;
	unsigned int count;
	struct analysis *left, *right;
};
struct analysis *head, *current;
struct analysis huff[512], temp;

// Function prototypes
void sort(int, int, int);
void buildtree(int, int, int);
void bitvalue(struct analysis *root, unsigned char *bit, unsigned char size);

// cuda function
__global__ void compress(unsigned char *input, unsigned int *offset, struct table_struct *d_table, unsigned char *temp, unsigned int filelength,unsigned int flag){
	unsigned int i, j, k;
	__shared__ struct  table_struct table;
	memcpy(&table, d_table, sizeof(struct table_struct));
	unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
	
	// when shared memory is sufficient
	if(flag == 0){
		for(i = pos; i < filelength; i += blockDim.x){
			for(k = 0; k < table.sizeDict[input[i]]; k++){
				temp[offset[i]+k] = table.bitDict[input[i]][k];
			}
		}
	}
	// use constant memory and shared memory
	else{
		for(i = pos; i < filelength; i += blockDim.x){
			for(k = 0; k < table.sizeDict[input[i]]; k++){
				if(k < 192)
					temp[offset[i]+k] = table.bitDict[input[i]][k];
				else
					temp[offset[i]+k] = d_bitDict[input[i]][k];
			}
		}
	}
	
	__syncthreads();
	
	for(i = pos * 8; i < offset[filelength]; i += blockDim.x * 8){
		for(j=0;j<8;j++){
			if(temp[i+j] == 0){
				input[i/8]=input[i/8] << 1;
			}
			else{
				input[i/8] = (input[i/8] << 1) | 1;
			}
		}
	}
}

int main(int argc, char **argv){
	unsigned int i, node = 0, arr = 0, filelength, frequency[256];
	FILE *source, *compressed;
	unsigned char *d_input, *h_input, *d_temp,  size = 0, bit[255];
	unsigned int *d_offset, *h_offset, cpu_time_used;
	struct table_struct *d_table;
	cudaError_t error;
	clock_t start, end;
	
	// start time measure
	start = clock();
	
	//open source and target compressed file
	source = fopen(argv[1], "rb");
	compressed = fopen(argv[2], "wb");
	
	//find length of source file
	fseeko(source, 0, SEEK_END);
	filelength = ftello(source);
	fseeko(source, 0, SEEK_SET);
	
	//allocate required memory and read the file to memory
	h_input = (unsigned char *)malloc(filelength * sizeof(unsigned char));
	fread(h_input, sizeof(unsigned char), filelength, source);
	fclose(source);
	
	//find the frequency of each symbols
	for (i = 0; i < 256; i++)
		frequency[i] = 0;
	for (i = 0; i < filelength; i++)
		frequency[h_input[i]]++;

	//initialize nodes of huffman tree
	for (i = 0; i < 256; i++){
		if (frequency[i] > 0)
		{
			node++;
			huff[node - 1].count = frequency[i];
			huff[node - 1].letter = i;
			huff[node - 1].left = NULL;
			huff[node - 1].right = NULL;
		}
	}
	
	//build tree 
	for (i = 0; i < node - 1; i++){
		arr = 2 * i;
		sort(i, node, arr);
		buildtree(i, node, arr);
	}
	
	//build table having the bit sequence and its length
	bitvalue(head, bit, size);

	//calculate h_offset
	h_offset = (unsigned int *)malloc((filelength + 1) * sizeof(unsigned int));
	h_offset[0] = 0;
	for(i = 0; i < filelength; i++){
		h_offset[i+1] = h_table.sizeDict[h_input[i]] + h_offset[i];
	}
	
	if(h_offset[filelength] % 8 != 0)
		h_offset[filelength] = h_offset[filelength] + (8 - (h_offset[filelength]%8));
	
	
	/////////////////END SERIAL///////////////////
	
	//////////////BEGIN PARALLEL//////////////////
	
	//malloc
	error = cudaMalloc((void **)&d_input, filelength * sizeof(unsigned char));
	if (error != cudaSuccess)
			printf("erro_1: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_offset, (filelength + 1) * sizeof(unsigned int));
	if (error != cudaSuccess)
			printf("erro_2: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_table, sizeof(table_struct));
	if (error != cudaSuccess)
			printf("erro_3: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_temp, h_offset[filelength] * sizeof(unsigned char));
	if (error!= cudaSuccess)
			printf("erro_5: %s\n", cudaGetErrorString(error));
	cudaMemcpyToSymbol (d_bitDict, max_bitDict, 256 * 255 * sizeof(unsigned char));
		
	//memcpy
	error = cudaMemcpy(d_input, h_input, filelength*sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
			printf("erro_6: %s\n", cudaGetErrorString(error));
	error = cudaMemcpy(d_offset, h_offset, (filelength + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
			printf("erro_7: %s\n", cudaGetErrorString(error));
	error = cudaMemcpy(d_table, &h_table, sizeof(table_struct), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
			printf("erro_8: %s\n", cudaGetErrorString(error));

	cudaMemset(d_temp, 0, h_offset[filelength] * sizeof(unsigned char));
	
	//run kernel and copy output
	//cudaDeviceSynchronize();
	compress<<<1, 1024>>>(d_input, d_offset, d_table, d_temp, filelength, flag);

	cudaMemcpy(h_input, d_input, ((h_offset[filelength] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaError_t error_final = cudaGetLastError();
	if (error_final != cudaSuccess)
		printf("erro_final: %s\n", cudaGetErrorString(error_final));
	
	cudaFree(d_input);
	cudaFree(d_offset);
	cudaFree(d_table);
	cudaFree(d_temp);
	
	//write the header to the file 
	fwrite(&filelength, sizeof(unsigned int), 1, compressed);
	fwrite(frequency, sizeof(unsigned int), 256, compressed);
	fwrite(h_input, sizeof(unsigned char), (h_offset[filelength] / 8), compressed);
	fclose(compressed);
	
	end = clock();
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("\ntime taken %d seconds and %d milliseconds\n\n", cpu_time_used / 1000, cpu_time_used % 1000);

	return 0;
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

//build tree based on sort result
void buildtree(int i, int node, int arr){
	free(head);
	head = (struct analysis *)malloc(sizeof(struct analysis));
	head->count = huff[arr].count + huff[arr + 1].count;
	head->left = &huff[arr];
	head->right = &huff[arr + 1];
	huff[node + i] = *head;
}

//get bit sequence for each char value
void bitvalue(struct analysis *root, unsigned char *bit, unsigned char size){
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
		h_table.sizeDict[root->letter] = size;
			
		if(size < 192){
			memcpy(h_table.bitDict[root->letter], bit, size * sizeof(unsigned char));
		}
		else{
			memcpy(max_bitDict[root->letter], bit, size * sizeof(unsigned char));
			flag = 1;
		}
	}
}

//function to print the tree
void printtree(struct analysis *tree){
	if (tree->left != NULL || tree->right != NULL){
		printtree(tree->left);
		printtree(tree->right);
	}
	else
		printf("%d\t%d\n", tree->letter, tree->count);
}
