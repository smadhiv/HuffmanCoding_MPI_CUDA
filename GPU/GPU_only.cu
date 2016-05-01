#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

struct table_struct
{
	unsigned char bit[60];
	unsigned char size;
}h_table[256];

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
void bitvalue(struct analysis *root, unsigned char *bit, unsigned char size);

__global__ void compress(unsigned char *input, unsigned int *offset, struct table_struct *table, unsigned char *temp, unsigned int filelength)
{
	unsigned int i, j, k;
	__shared__ struct table_struct d_table[256];
	memcpy(d_table, table, 256*sizeof(struct table_struct));
	unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
	
	for(i = pos; i < filelength; i += blockDim.x)
	{
		for(k = 0; k < d_table[input[i]].size; k++)
		{
			temp[offset[i]+k] = d_table[input[i]].bit[k];
		}
	}
	
	__syncthreads();
	
	for(i = pos * 8; i < offset[filelength]; i += blockDim.x * 8)
	{
		for(j=0;j<8;j++)
		{
			if(temp[i+j] == 0)
			{
				input[i/8]=input[i/8] << 1;
			}
			else
			{
				input[i/8] = (input[i/8] << 1) | 1;
			}
		}
	}
	__syncthreads();
}


__global__ void getFrequency(unsigned char *input, unsigned int *frequency, unsigned int filelength)
{
	unsigned int i;
	__shared__ unsigned int freq[256];
	memset(freq, 0, 256* 4);
	unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
//	if(pos < 256){
	//	freq[pos] = 0;
//}
	__syncthreads();
	for(i = pos; i < filelength; i += blockDim.x)
	{
		atomicAdd(&freq[input[i]], 1);
	}
	/*
	for(i = 0; i < filelength; i++)
	{}
		if(input[i] == pos){
			freq[pos]++;
		}
	}*/
	__syncthreads();
	if(pos < 256){
		memcpy(&frequency[pos], &freq[pos], sizeof(unsigned int));
	}
	
}

int main(int argc, char **argv)
{
	unsigned int i, node = 0, arr = 0, filelength, frequency[256], *d_frequency;
	FILE *source, *compressed;
	unsigned char *d_input, *h_input, *d_temp,  size = 0, bit[100];
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
	h_input = (unsigned char*)malloc(filelength*sizeof(unsigned char));
	fread(h_input, sizeof(unsigned char), filelength, source);
	fclose(source);
	
	error = cudaMalloc((void **)&d_input, filelength*sizeof(unsigned char));
	if (error != cudaSuccess)
			printf("erro_1: %s\n", cudaGetErrorString(error));
	// Get frequency
	error = cudaMalloc((void **)&d_frequency, 256*sizeof(unsigned int));
	if (error != cudaSuccess)
			printf("k1_erro_1: %s\n", cudaGetErrorString(error));
	error = cudaMemcpy(d_input, h_input, filelength*sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
				printf("erro_6: %s\n", cudaGetErrorString(error));
	getFrequency<<<2, 512>>>(d_input, d_frequency, filelength);
	cudaDeviceSynchronize();

	//find the frequency of each symbols
	cudaMemcpy(frequency, d_frequency, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	
	cudaFree(d_input);
	cudaFree(d_frequency);
	/*for (i = 0; i < 256; i++)
		frequency[i] = 0;
	for (i = 0; i < filelength; i++)
		frequency[h_input[i]]++;*/

	//initialize nodes of huffman tree
	for (i = 0; i < 256; i++)
		if (frequency[i] > 0)
		{
			node++;
			huff[node - 1].count = frequency[i];
			huff[node - 1].letter = i;
			huff[node - 1].left = NULL;
			huff[node - 1].right = NULL;
		}

	//build tree 
	for (i = 0; i < node - 1; i++)
	{
		arr = 2 * i;
		sort(i, node, arr);
		buildtree(i, node, arr);
	}
	
	//build table having the bit sequence and its length
	bitvalue(head, bit, size);

	//calculate h_offset
	h_offset = (unsigned int *)malloc((filelength + 1)*sizeof(unsigned int));
	h_offset[0] = 0;
	for(i = 0; i < filelength; i++)
	{
		h_offset[i+1] = h_table[h_input[i]].size + h_offset[i];
	}
	
	if(h_offset[filelength]%8!=0)
		h_offset[filelength] = h_offset[filelength] + (8 - (h_offset[filelength]%8));
	
	
	/////////////////END SERIAL///////////////////
	
	//////////////BEGIN PARALLEL//////////////////
	
	
	error = cudaMalloc((void **)&d_input, filelength*sizeof(unsigned char));
	if (error != cudaSuccess)
			printf("erro_1: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_offset, (filelength + 1)*sizeof(unsigned int));
	if (error != cudaSuccess)
			printf("erro_3: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_table, 256*sizeof(table_struct));
	if (error != cudaSuccess)
			printf("erro_4: %s\n", cudaGetErrorString(error));
	error = cudaMalloc((void **)&d_temp, h_offset[filelength]*sizeof(unsigned char));
	cudaMemset(d_temp, 0, h_offset[filelength]*sizeof(unsigned char));
	if (error!= cudaSuccess)
			printf("erro_5: %s\n", cudaGetErrorString(error));
	
	printf("Total GPU space: %.3fMB\n", (filelength*sizeof(unsigned char) +
										(h_offset[filelength]/8)*sizeof(unsigned char) +
										(filelength + 1)*sizeof(unsigned int) +
										256*sizeof(table_struct) +
										h_offset[filelength]*sizeof(unsigned char))/1000000.0);
	
	error = cudaMemcpy(d_input, h_input, filelength*sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
				printf("erro_6: %s\n", cudaGetErrorString(error));
	error = cudaMemcpy(d_offset, h_offset, (filelength + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
				printf("erro_7: %s\n", cudaGetErrorString(error));
	error = cudaMemcpy(d_table, h_table, 256 * sizeof(table_struct), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
				printf("erro_8: %s\n", cudaGetErrorString(error));
	
	//run kernel and copy output
	compress<<<1, 1024>>>(d_input, d_offset, d_table, d_temp, filelength);
	cudaMemcpy(h_input, d_input, ((h_offset[filelength]/8))*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	//print d_table
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
	fwrite(h_input, sizeof(unsigned char), (h_offset[filelength]/8), compressed);
	fclose(compressed);
	
	end = clock();
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("\ntime taken %d seconds and %d milliseconds\n\n", cpu_time_used / 1000, cpu_time_used % 1000);

	return 0;
}

// sort nodes based on frequency
void sort(int i, int node, int arr)
{
	int a, b;
	for (a = arr; a < node - 1 + i; a++)
		for (b = arr; b < node - 1 + i; b++)
			if (huff[b].count > huff[b + 1].count)
			{
				temp = huff[b];
				huff[b] = huff[b + 1];
				huff[b + 1] = temp;
			}
}

//build tree based on sort result
void buildtree(int i, int node, int arr)
{
	free(head);
	head = (struct analysis *)malloc(sizeof(struct analysis));
	head->count = huff[arr].count + huff[arr + 1].count;
	head->left = &huff[arr];
	head->right = &huff[arr + 1];
	huff[node + i] = *head;
}

//get bit sequence for each char value
void bitvalue(struct analysis *root, unsigned char *bit, unsigned char size)
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
		h_table[root->letter].size = size;
		memcpy(h_table[root->letter].bit, bit, size*sizeof(unsigned char));
		/*
		printf("\nchar %c\t size %d\n", root->letter, size);
		for (i = 0; i < size;i++)
		{
		printf("%d", table[root->letter].bit[i]);
		}
		*/
	}
}

//function to print the tree
void printtree(struct analysis *tree)
{
	if (tree->left != NULL || tree->right != NULL)
	{
		printtree(tree->left);
		printtree(tree->right);
	}
	else
		printf("%d\t%d\n", tree->letter, tree->count);
}
