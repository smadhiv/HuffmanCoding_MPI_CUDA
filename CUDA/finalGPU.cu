//Sriram Madhivanan
//Struct of Arrays
//Constant memory if dictinary goes beyond 191 bits
//Max possible shared memory
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

//4294967295

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
__global__ void compress(unsigned char *d_input, unsigned int *d_offset, struct table_struct *d_table, unsigned char *d_temp, unsigned int d_filelength, unsigned int flag){
	__shared__ struct  table_struct table;
	memcpy(&table, d_table, sizeof(struct table_struct));
	unsigned int filelength = d_filelength;
	unsigned int i, j, k;
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
	
	// when shared memory is sufficient
	if(flag == 0){
		for(i = pos; i < filelength; i += blockDim.x){
			for(k = 0; k < table.sizeDict[d_input[i]]; k++){
				d_temp[d_offset[i]+k] = table.bitDict[d_input[i]][k];
			}
		}
	}
	// use constant memory and shared memory
	else{
		for(i = pos; i < filelength; i += blockDim.x){
			for(k = 0; k < table.sizeDict[d_input[i]]; k++){
				if(k < 191)
					d_temp[d_offset[i]+k] = table.bitDict[d_input[i]][k];
				else
					d_temp[d_offset[i]+k] = d_bitDict[d_input[i]][k];
			}
		}
	}
	__syncthreads();
	
	for(i = pos * 8; i < d_offset[filelength]; i += blockDim.x * 8){
		for(j = 0; j < 8; j++){
			if(d_temp[i + j] == 0){
				d_input[i / 8] = d_input[i / 8] << 1;
			}
			else{
				d_input[i / 8] = (d_input[i / 8] << 1) | 1;
			}
		}
	}
	__syncthreads();
}

// cuda function
__global__ void compressOverflow(unsigned char *d_input, unsigned int *d_offset, struct table_struct *d_table, unsigned char *d_temp, unsigned char *d_temp_overflow, unsigned int d_filelength, unsigned int flag, unsigned int overflowPosition){
	__shared__ struct  table_struct table;
	memcpy(&table, d_table, sizeof(struct table_struct));
	unsigned int filelength = d_filelength;
	unsigned int i, j, k;
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int offset_overflow;
	
	// when shared memory is sufficient
	if(flag == 0){
		for(i = pos; i < overflowPosition; i += blockDim.x){
			for(k = 0; k < table.sizeDict[d_input[i]]; k++){
				d_temp[d_offset[i]+k] = table.bitDict[d_input[i]][k];
			}
		}
		for(i = overflowPosition + pos; i < filelength - 1; i += blockDim.x){
			for(k = 0; k < table.sizeDict[d_input[i + 1]]; k++){
				d_temp_overflow[d_offset[i + 1] + k] = table.bitDict[d_input[i + 1]][k];
			}
		}
		if(pos == 0){
			memcpy(&d_temp_overflow[d_offset[(overflowPosition + 1)] - table.sizeDict[d_input[overflowPosition]]], &table.bitDict[d_input[overflowPosition]], table.sizeDict[d_input[overflowPosition]]);
		}
	}
	// use constant memory and shared memory
	else{
		for(i = pos; i < filelength; i += blockDim.x){
			for(k = 0; k < table.sizeDict[d_input[i]]; k++){
				if(k < 191)
					d_temp[d_offset[i]+k] = table.bitDict[d_input[i]][k];
				else
					d_temp[d_offset[i]+k] = d_bitDict[d_input[i]][k];
			}
		}
	}
	__syncthreads();
	
	for(i = pos * 8; i < d_offset[overflowPosition]; i += blockDim.x * 8){
		for(j = 0; j < 8; j++){
			if(d_temp[i + j] == 0){
				d_input[i / 8] = d_input[i / 8] << 1;
			}
			else{
				d_input[i / 8] = (d_input[i / 8] << 1) | 1;
			}
		}
	}
	offset_overflow = d_offset[overflowPosition] / 8;
	
	for(i = pos * 8; i < d_offset[filelength]; i += blockDim.x * 8){
		for(j = 0; j < 8; j++){
			if(d_temp_overflow[i + j] == 0){
				d_input[(i / 8) + offset_overflow] = d_input[(i / 8) + offset_overflow] << 1;
			}
			else{
				d_input[(i / 8) + offset_overflow] = (d_input[(i / 8) + offset_overflow] << 1) | 1;
			}
		}
	}
}

int main(int argc, char **argv){
	unsigned int i, node = 0, arr = 0, filelength, frequency[256];
	FILE *sourceFile, *compressedFile;
	unsigned char *d_input, *h_input, *d_temp,  size = 0, bit[255];
	unsigned int *d_offset, *h_offset, cpu_time_used;
	struct table_struct *d_table;
	unsigned int flagOverflow, overflowPosition, flagPadding;
	cudaError_t error;
	clock_t start, end;
	
	// start time measure
	start = clock();
	
	// read input file, get filelength and data
	sourceFile = fopen(argv[1], "rb");
	fseek(sourceFile, 0, SEEK_END);
	filelength = ftell(sourceFile);
	fseek(sourceFile, 0, SEEK_SET);
	h_input = (unsigned char *)malloc(filelength * sizeof(unsigned char));
	fread(h_input, sizeof(unsigned char), filelength, sourceFile);
	fclose(sourceFile);
	
	// find the frequency of each symbols
	for (i = 0; i < 256; i++){
		frequency[i] = 0;
	}
	for (i = 0; i < filelength; i++){
		frequency[h_input[i]]++;
	}

	// initialize nodes of huffman tree
	for (i = 0; i < 256; i++){
		if (frequency[i] > 0){
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

	// calculate h_offset
	flagOverflow = 0;
	flagPadding = 0;
	h_offset = (unsigned int *)malloc((filelength + 1) * sizeof(unsigned int));
	h_offset[0] = 0;
	for(i = 0; i < filelength; i++){
		h_offset[i + 1] = h_table.sizeDict[h_input[i]] + h_offset[i];
		if(h_offset[i + 1] + 1048576 < h_offset[i]){
			printf("Overflow error Occured\n");
			flagOverflow = 1;
			overflowPosition = i;
			if(h_offset[i] % 8 != 0){
				flagPadding = 1;
				h_offset[i + 1] = (h_offset[i] % 8) + h_table.sizeDict[h_input[i]];
				h_offset[i] = h_offset[i] + (8 - (h_offset[i] % 8));
			}
			else{
				h_offset[i + 1] = 0;				
			}
		}
	}
	if(h_offset[filelength] % 8 != 0){
		h_offset[filelength] = h_offset[filelength] + (8 - (h_offset[filelength] % 8));
	}

	for(i = overflowPosition; i < overflowPosition + 20; i ++){
		printf("%u\t%u\n", h_offset[i + 1], h_table.sizeDict[h_input[i + 1]]);
	}

	/////////////////END SERIAL///////////////////
	
	//////////////BEGIN PARALLEL//////////////////
	

	if(flagOverflow == 0){
		long unsigned int mem_free, mem_total;
		long unsigned int mem_req;
		mem_req = 2 + (filelength * sizeof(unsigned char) + (filelength + 1) * sizeof(unsigned int) + sizeof(table_struct) + (long unsigned int)h_offset[filelength] * sizeof(unsigned char))/(1024 * 1024);
		printf("Total GPU space required: %lu\n", mem_req);

		// query device memory
		error = cudaMemGetInfo(&mem_free, &mem_total);
		printf("Total GPU memory: %lu\n", mem_total/(1024 * 1024));
		printf("Total GPU space available: %lu\n", mem_free/(1024 * 1024));
			
		if(mem_req < mem_free){
			// malloc
			error = cudaMalloc((void **)&d_input, filelength * sizeof(unsigned char));
			if (error != cudaSuccess)
					printf("erro_1: %s\n", cudaGetErrorString(error));
			error = cudaMalloc((void **)&d_offset, (filelength + 1) * sizeof(unsigned int));
			if (error != cudaSuccess)
					printf("erro_2: %s\n", cudaGetErrorString(error));
			error = cudaMalloc((void **)&d_table, sizeof(table_struct));
			if (error != cudaSuccess)
					printf("erro_3: %s\n", cudaGetErrorString(error));
			error = cudaMalloc((void **)&d_temp, (h_offset[filelength]) * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_5: %s\n", cudaGetErrorString(error));
	
			// memcpy
			error = cudaMemcpy(d_input, h_input, filelength * sizeof(unsigned char), cudaMemcpyHostToDevice);
			if (error!= cudaSuccess)
					printf("erro_6: %s\n", cudaGetErrorString(error));
			error = cudaMemcpy(d_offset, h_offset, (filelength + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
			if (error!= cudaSuccess)
					printf("erro_7: %s\n", cudaGetErrorString(error));
			error = cudaMemcpy(d_table, &h_table, sizeof(table_struct), cudaMemcpyHostToDevice);
			if (error!= cudaSuccess)
					printf("erro_8: %s\n", cudaGetErrorString(error));
			// initialize d_temp 
			error = cudaMemset(d_temp, 0, h_offset[filelength] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_9: %s\n", cudaGetErrorString(error));		
			// copy constant memory
			//if(flag == 1){
			error = cudaMemcpyToSymbol (d_bitDict, max_bitDict, 256 * 255 * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_10: %s\n", cudaGetErrorString(error));
			//}
			
	
			// run kernel and copy output
			error = cudaMemGetInfo(&mem_free, &mem_total);
			printf("Total GPU memory: %lu\n", mem_total/(1024 * 1024));
			printf("Total GPU space available: %lu\n", mem_free/(1024 * 1024));
		
			compress<<<1, 1024>>>(d_input, d_offset, d_table, d_temp, filelength, flag);
			cudaError_t error_kernel = cudaGetLastError();
			if (error_kernel != cudaSuccess)
				printf("erro_final: %s\n", cudaGetErrorString(error_kernel));

			error = cudaMemcpy(h_input, d_input, ((h_offset[filelength] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			if (error != cudaSuccess)
				printf("erro_11: %s\n", cudaGetErrorString(error));
			printf("%lu\n", ((h_offset[filelength] / 8)) * sizeof(unsigned char));
			
			cudaFree(d_input);
			cudaFree(d_offset);
			cudaFree(d_table);
			cudaFree(d_temp);
			
			// write src filelength, header and compressed data to output file
			compressedFile = fopen(argv[2], "wb");
			fwrite(&filelength, sizeof(unsigned int), 1, compressedFile);
			fwrite(frequency, sizeof(unsigned int), 256, compressedFile);
			fwrite(h_input, sizeof(unsigned char), (h_offset[filelength] / 8), compressedFile);
			fclose(compressedFile);			
		}
	}
	else{
		//printf("Input File Size = %u Bytes\nh_offset[filelength] = %lu\nOutput File Size = %u\n", filelength, (long unsigned int)((long unsigned int)h_offset[filelength] + (long unsigned int)h_offset[overflowPosition]), ((h_offset[filelength]/8) + (h_offset[overflowPosition]/8)));
		printf("overflowPosition - 1 = %u\n", h_offset[overflowPosition - 1]);
		printf("overflowPosition     = %u\n", h_offset[overflowPosition]);
		printf("overflowPosition + 1 = %u\n", h_offset[overflowPosition + 1]);
		printf("overflowPosition + 2 = %u\n", h_offset[overflowPosition + 2]);
		printf("h_offset[filelength] = %u\n", h_offset[filelength]);
		printf("overflowPosition is %u\n", overflowPosition);
		printf("filelength       is %u\n", filelength);		
		printf("flag             is %u\n", flag);
		//for(i = overflowPosition; i < overflowPosition + 10; i++){
		//		printf("%u\t%u\n",h_offset[i + 1], h_table.sizeDict[h_input[i + 1]]);
		//}
		//for(i = 0; i < 10; i++){
		//		printf("%u\t%u\n",h_offset[i + 1], h_table.sizeDict[h_input[i + 1]]);
		//}
		long unsigned int mem_free, mem_total;
		long unsigned int mem_req;
		mem_req = 2 + (long unsigned int)((long unsigned int)filelength * sizeof(unsigned char) + (long unsigned int)(filelength + 1) * sizeof(unsigned int) + sizeof(table_struct) + (long unsigned int)h_offset[overflowPosition] * sizeof(unsigned char) + (long unsigned int)h_offset[filelength] * sizeof(unsigned char))/(1024 * 1024);
		printf("Total GPU space required: %lu\n", mem_req);

		// query device memory
		error = cudaMemGetInfo(&mem_free, &mem_total);
		printf("Total GPU memory: %lu\n", mem_total/(1024 * 1024));
		printf("Total GPU space available: %lu\n", mem_free/(1024 * 1024));
			
		if(mem_req < mem_free){
			unsigned char *d_temp_overflow;
			// malloc
			
			// allocate input file data
			error = cudaMalloc((void **)&d_input, filelength * sizeof(unsigned char));
			if (error != cudaSuccess)
					printf("erro_1: %s\n", cudaGetErrorString(error));
				
			// allocate offset 
			error = cudaMalloc((void **)&d_offset, (filelength + 1) * sizeof(unsigned int));
			if (error != cudaSuccess)
					printf("erro_2: %s\n", cudaGetErrorString(error));
				
			// allocate structure
			error = cudaMalloc((void **)&d_table, sizeof(table_struct));
			if (error != cudaSuccess)
					printf("erro_3: %s\n", cudaGetErrorString(error));
				
			// allocate bit to byte storage
			error = cudaMalloc((void **)&d_temp, h_offset[overflowPosition] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_5: %s\n", cudaGetErrorString(error));
				
			error = cudaMalloc((void **)&d_temp_overflow, h_offset[filelength] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_6: %s\n", cudaGetErrorString(error));
							
			// memcpy
			// copy input data
			error = cudaMemcpy(d_input, h_input, filelength * sizeof(unsigned char), cudaMemcpyHostToDevice);
			if (error!= cudaSuccess)
					printf("erro_7: %s\n", cudaGetErrorString(error));
				
			// copy offset
			error = cudaMemcpy(d_offset, h_offset, (filelength + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
			if (error!= cudaSuccess)
					printf("erro_8: %s\n", cudaGetErrorString(error));
				
			// copy structure
			error = cudaMemcpy(d_table, &h_table, sizeof(table_struct), cudaMemcpyHostToDevice);
			if (error!= cudaSuccess)
					printf("erro_9: %s\n", cudaGetErrorString(error));
			
			// initialize d_temp
			// initialize bit to byte array to  0
			error = cudaMemset(d_temp, 0, h_offset[overflowPosition] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_10: %s\n", cudaGetErrorString(error));	
				
			error = cudaMemset(d_temp_overflow, 0, h_offset[filelength] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_11: %s\n", cudaGetErrorString(error));		
				
			// copy constant memory
			if(flag == 1){
				error = cudaMemcpyToSymbol (d_bitDict, max_bitDict, 256 * 255 * sizeof(unsigned char));
				if (error!= cudaSuccess)
					printf("erro_12: %s\n", cudaGetErrorString(error));
			}
			
	
			// get GPU storage data after transfers
			error = cudaMemGetInfo(&mem_free, &mem_total);
			printf("Total GPU memory: %lu\n", mem_total/(1024 * 1024));
			printf("Total GPU space available: %lu\n", mem_free/(1024 * 1024));
			
			// launch kernel
			compressOverflow<<<1, 1024>>>(d_input, d_offset, d_table, d_temp, d_temp_overflow, filelength, flag, overflowPosition);
			
			// check status
			cudaError_t error_kernel = cudaGetLastError();
			if (error_kernel != cudaSuccess)
				printf("erro_final: %s\n", cudaGetErrorString(error_kernel));
			
			// get output data
			if(flagPadding == 0){
				error = cudaMemcpy(h_input, d_input, (h_offset[overflowPosition] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("erro_11: %s\n", cudaGetErrorString(error));
				error = cudaMemcpy(&h_input[(h_offset[overflowPosition] / 8)], &d_input[(h_offset[overflowPosition] / 8)], ((h_offset[filelength] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("erro_12: %s\n", cudaGetErrorString(error));
			}
			else{
				printf("In scary zone\n");
				error = cudaMemcpy(h_input, d_input, (h_offset[overflowPosition] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("erro_11: %s\n", cudaGetErrorString(error));
				unsigned char temp = h_input[(h_offset[overflowPosition] / 8) - 1];
				
				error = cudaMemcpy(&h_input[(h_offset[overflowPosition] / 8) - 1], &d_input[(h_offset[overflowPosition] / 8)], ((h_offset[filelength] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("erro_12: %s\n", cudaGetErrorString(error));				
				h_input[(h_offset[overflowPosition] / 8) - 1] = temp | h_input[(h_offset[overflowPosition] / 8) - 1];
			}

			cudaFree(d_input);
			cudaFree(d_offset);
			cudaFree(d_table);
			cudaFree(d_temp);
			cudaFree(d_temp_overflow);
			
			// write src filelength, header and compressed data to output file
			compressedFile = fopen(argv[2], "wb");
			fwrite(&filelength, sizeof(unsigned int), 1, compressedFile);
			fwrite(frequency, sizeof(unsigned int), 256, compressedFile);
			fwrite(h_input, sizeof(unsigned char), (h_offset[filelength] / 8 + h_offset[overflowPosition] / 8) - 1, compressedFile);
			fclose(compressedFile);			
		}
	}
	// calculate run duration
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
void bitvalue(struct analysis *root, unsigned char *bit, unsigned char size){
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
