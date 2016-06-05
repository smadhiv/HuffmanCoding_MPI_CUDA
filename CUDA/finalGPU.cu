//Sriram Madhivanan
//Struct of Arrays
//Constant memory if dictinary goes beyond 191 bits
//Max possible shared memory
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

struct huffmanDictionary
{
	unsigned char bitSequence[256][191];
	unsigned char bitSequenceLength[256];
}huffmanDictionary;

struct huffmanTree
{
	unsigned char letter;
	unsigned int count;
	struct huffmanTree *left, *right;
};
struct huffmanTree *head_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512], temp_huffmanTreeNode;

// handles when constant memory is needed to access bit sequence of length > 191
__constant__ unsigned char d_bitSequenceConstMemory[256][255];
unsigned char bitSequenceConstMemory[256][255];
unsigned int constMemoryFlag = 0;

// Function prototypes
void sortHuffmanTree(int, int, int);
void buildHuffmanTree(int, int, int);
void buildHuffmanDictionary(struct huffmanTree *, unsigned char *, unsigned char);

// cuda function
__global__ void compress(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, struct huffmanDictionary *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned int d_inputFileLength, unsigned int constMemoryFlag){
	__shared__ struct  huffmanDictionary table;
	memcpy(&table, d_huffmanDictionary, sizeof(struct huffmanDictionary));
	unsigned int inputFileLength = d_inputFileLength;
	unsigned int i, j, k;
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
	
	// when shared memory is sufficient
	if(constMemoryFlag == 0){
		for(i = pos; i < inputFileLength; i += blockDim.x){
			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i]]; k++){
				d_byteCompressedData[d_compressedDataOffset[i]+k] = table.bitSequence[d_inputFileData[i]][k];
			}
		}
	}
	// use constant memory and shared memory
	else{
		for(i = pos; i < inputFileLength; i += blockDim.x){
			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i]]; k++){
				if(k < 191)
					d_byteCompressedData[d_compressedDataOffset[i]+k] = table.bitSequence[d_inputFileData[i]][k];
				else
					d_byteCompressedData[d_compressedDataOffset[i]+k] = d_bitSequenceConstMemory[d_inputFileData[i]][k];
			}
		}
	}
	__syncthreads();
	
	for(i = pos * 8; i < d_compressedDataOffset[inputFileLength]; i += blockDim.x * 8){
		for(j = 0; j < 8; j++){
			if(d_byteCompressedData[i + j] == 0){
				d_inputFileData[i / 8] = d_inputFileData[i / 8] << 1;
			}
			else{
				d_inputFileData[i / 8] = (d_inputFileData[i / 8] << 1) | 1;
			}
		}
	}
	__syncthreads();
}

// cuda function
__global__ void compress(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, struct huffmanDictionary *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned char *d_temp_overflow, unsigned int d_inputFileLength, unsigned int constMemoryFlag, unsigned int overflowPosition){
	__shared__ struct  huffmanDictionary table;
	memcpy(&table, d_huffmanDictionary, sizeof(struct huffmanDictionary));
	unsigned int inputFileLength = d_inputFileLength;
	unsigned int i, j, k;
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int offset_overflow;
	
	// when shared memory is sufficient
	if(constMemoryFlag == 0){
		for(i = pos; i < overflowPosition; i += blockDim.x){
			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i]]; k++){
				d_byteCompressedData[d_compressedDataOffset[i]+k] = table.bitSequence[d_inputFileData[i]][k];
			}
		}
		for(i = overflowPosition + pos; i < inputFileLength - 1; i += blockDim.x){
			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i + 1]]; k++){
				d_temp_overflow[d_compressedDataOffset[i + 1] + k] = table.bitSequence[d_inputFileData[i + 1]][k];
			}
		}
		if(pos == 0){
			memcpy(&d_temp_overflow[d_compressedDataOffset[(overflowPosition + 1)] - table.bitSequenceLength[d_inputFileData[overflowPosition]]], &table.bitSequence[d_inputFileData[overflowPosition]], table.bitSequenceLength[d_inputFileData[overflowPosition]]);
		}
	}
	// use constant memory and shared memory
	else{
		for(i = pos; i < inputFileLength; i += blockDim.x){
			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i]]; k++){
				if(k < 191)
					d_byteCompressedData[d_compressedDataOffset[i]+k] = table.bitSequence[d_inputFileData[i]][k];
				else
					d_byteCompressedData[d_compressedDataOffset[i]+k] = d_bitSequenceConstMemory[d_inputFileData[i]][k];
			}
		}
	}
	__syncthreads();
	
	for(i = pos * 8; i < d_compressedDataOffset[overflowPosition]; i += blockDim.x * 8){
		for(j = 0; j < 8; j++){
			if(d_byteCompressedData[i + j] == 0){
				d_inputFileData[i / 8] = d_inputFileData[i / 8] << 1;
			}
			else{
				d_inputFileData[i / 8] = (d_inputFileData[i / 8] << 1) | 1;
			}
		}
	}
	offset_overflow = d_compressedDataOffset[overflowPosition] / 8;
	
	for(i = pos * 8; i < d_compressedDataOffset[inputFileLength]; i += blockDim.x * 8){
		for(j = 0; j < 8; j++){
			if(d_temp_overflow[i + j] == 0){
				d_inputFileData[(i / 8) + offset_overflow] = d_inputFileData[(i / 8) + offset_overflow] << 1;
			}
			else{
				d_inputFileData[(i / 8) + offset_overflow] = (d_inputFileData[(i / 8) + offset_overflow] << 1) | 1;
			}
		}
	}
}

int main(int argc, char **argv){
	unsigned int i;
	unsigned int distinctCharacterCount, combinedHuffmanNodes, inputFileLength, frequency[256];
	unsigned char *d_inputFileData, *inputFileData, *d_byteCompressedData,  bitSequenceLength = 0, bitSequence[255];
	unsigned int *d_compressedDataOffset, *compressedDataOffset, cpu_time_used;
	struct huffmanDictionary *d_huffmanDictionary;
	unsigned int integerOverflowFlag, integerOverflowIndex, bitPaddingFlag;
	FILE *inputFile, *compressedFile;
	cudaError_t error;
	clock_t start, end;
	
	// start time measure
	start = clock();
	
	// read input file, get inputFileLength and data
	inputFile = fopen(argv[1], "rb");
	fseek(inputFile, 0, SEEK_END);
	inputFileLength = ftell(inputFile);
	fseek(inputFile, 0, SEEK_SET);
	inputFileData = (unsigned char *)malloc(inputFileLength * sizeof(unsigned char));
	fread(inputFileData, sizeof(unsigned char), inputFileLength, inputFile);
	fclose(inputFile);
	
	// find the frequency of each symbols
	for (i = 0; i < 256; i++){
		frequency[i] = 0;
	}
	for (i = 0; i < inputFileLength; i++){
		frequency[inputFileData[i]]++;
	}

	// initialize nodes of huffman tree
	distinctCharacterCount = 0;
	for (i = 0; i < 256; i++){
		if (frequency[i] > 0){
			huffmanTreeNode[distinctCharacterCount].count = frequency[i];
			huffmanTreeNode[distinctCharacterCount].letter = i;
			huffmanTreeNode[distinctCharacterCount].left = NULL;
			huffmanTreeNode[distinctCharacterCount].right = NULL;
			distinctCharacterCount++;
		}
	}
	
	// build tree 
	for (i = 0; i < distinctCharacterCount - 1; i++){
		combinedHuffmanNodes = 2 * i;
		sortHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
		buildHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
	}
	
	// build table having the bitSequence sequence and its length
	buildHuffmanDictionary(head_huffmanTreeNode, bitSequence, bitSequenceLength);

	// calculate compressed data offset - (1048576 is a safe number that will ensure there is no integer overflow in GPU, it should be minimum 8 * number of threads)
	integerOverflowFlag = 0;
	bitPaddingFlag = 0;
	compressedDataOffset = (unsigned int *)malloc((inputFileLength + 1) * sizeof(unsigned int));
	compressedDataOffset[0] = 0;
	for(i = 0; i < inputFileLength; i++){
		compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]] + compressedDataOffset[i];
		if(compressedDataOffset[i + 1] + 1048576 < compressedDataOffset[i]){
			printf("Overflow error Occured\n");
			integerOverflowFlag = 1;
			integerOverflowIndex = i;
			if(compressedDataOffset[i] % 8 != 0){
				bitPaddingFlag = 1;
				compressedDataOffset[i + 1] = (compressedDataOffset[i] % 8) + huffmanDictionary.bitSequenceLength[inputFileData[i]];
				compressedDataOffset[i] = compressedDataOffset[i] + (8 - (compressedDataOffset[i] % 8));
			}
			else{
				compressedDataOffset[i + 1] = 0;				
			}
		}
	}
	if(compressedDataOffset[inputFileLength] % 8 != 0){
		compressedDataOffset[inputFileLength] = compressedDataOffset[inputFileLength] + (8 - (compressedDataOffset[inputFileLength] % 8));
	}

	if(integerOverflowFlag == 0){
		long unsigned int mem_free, mem_total;
		long unsigned int mem_req;
		mem_req = 2 + (inputFileLength * sizeof(unsigned char) + (inputFileLength + 1) * sizeof(unsigned int) + sizeof(huffmanDictionary) + (long unsigned int)compressedDataOffset[inputFileLength] * sizeof(unsigned char))/(1024 * 1024);
		printf("Total GPU space required: %lu\n", mem_req);

		// query device memory
		error = cudaMemGetInfo(&mem_free, &mem_total);
		printf("Total GPU memory: %lu\n", mem_total/(1024 * 1024));
		printf("Total GPU space available: %lu\n", mem_free/(1024 * 1024));
			
		if(mem_req < mem_free){
			// malloc
			error = cudaMalloc((void **)&d_inputFileData, inputFileLength * sizeof(unsigned char));
			if (error != cudaSuccess)
					printf("erro_1: %s\n", cudaGetErrorString(error));
			error = cudaMalloc((void **)&d_compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int));
			if (error != cudaSuccess)
					printf("erro_2: %s\n", cudaGetErrorString(error));
			error = cudaMalloc((void **)&d_huffmanDictionary, sizeof(huffmanDictionary));
			if (error != cudaSuccess)
					printf("erro_3: %s\n", cudaGetErrorString(error));
			error = cudaMalloc((void **)&d_byteCompressedData, (compressedDataOffset[inputFileLength]) * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_5: %s\n", cudaGetErrorString(error));
	
			// memcpy
			error = cudaMemcpy(d_inputFileData, inputFileData, inputFileLength * sizeof(unsigned char), cudaMemcpyHostToDevice);
			if (error!= cudaSuccess)
					printf("erro_6: %s\n", cudaGetErrorString(error));
			error = cudaMemcpy(d_compressedDataOffset, compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
			if (error!= cudaSuccess)
					printf("erro_7: %s\n", cudaGetErrorString(error));
			error = cudaMemcpy(d_huffmanDictionary, &huffmanDictionary, sizeof(huffmanDictionary), cudaMemcpyHostToDevice);
			if (error!= cudaSuccess)
					printf("erro_8: %s\n", cudaGetErrorString(error));
				
			// initialize d_byteCompressedData 
			error = cudaMemset(d_byteCompressedData, 0, compressedDataOffset[inputFileLength] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_9: %s\n", cudaGetErrorString(error));
				
			// copy constant memory
			if(constMemoryFlag == 1){
				error = cudaMemcpyToSymbol (d_bitSequenceConstMemory, bitSequenceConstMemory, 256 * 255 * sizeof(unsigned char));
				if (error!= cudaSuccess)
					printf("erro_10: %s\n", cudaGetErrorString(error));
			}
			
			// run kernel and copy output
			error = cudaMemGetInfo(&mem_free, &mem_total);
			printf("Total GPU memory: %lu\n", mem_total/(1024 * 1024));
			printf("Total GPU space available: %lu\n", mem_free/(1024 * 1024));
		
			compress<<<1, 1024>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, inputFileLength, constMemoryFlag);
			cudaError_t error_kernel = cudaGetLastError();
			if (error_kernel != cudaSuccess)
				printf("erro_final: %s\n", cudaGetErrorString(error_kernel));

			error = cudaMemcpy(inputFileData, d_inputFileData, ((compressedDataOffset[inputFileLength] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			if (error != cudaSuccess)
				printf("erro_11: %s\n", cudaGetErrorString(error));
			printf("%lu\n", ((compressedDataOffset[inputFileLength] / 8)) * sizeof(unsigned char));
			
			cudaFree(d_inputFileData);
			cudaFree(d_compressedDataOffset);
			cudaFree(d_huffmanDictionary);
			cudaFree(d_byteCompressedData);
			
			// write src inputFileLength, header and compressed data to output file
			compressedFile = fopen(argv[2], "wb");
			fwrite(&inputFileLength, sizeof(unsigned int), 1, compressedFile);
			fwrite(frequency, sizeof(unsigned int), 256, compressedFile);
			fwrite(inputFileData, sizeof(unsigned char), (compressedDataOffset[inputFileLength] / 8), compressedFile);
			fclose(compressedFile);			
		}
	}
	else{
		long unsigned int mem_free, mem_total;
		long unsigned int mem_req;
		mem_req = 2 + (long unsigned int)((long unsigned int)inputFileLength * sizeof(unsigned char) + (long unsigned int)(inputFileLength + 1) * sizeof(unsigned int) + sizeof(huffmanDictionary) + (long unsigned int)compressedDataOffset[integerOverflowIndex] * sizeof(unsigned char) + (long unsigned int)compressedDataOffset[inputFileLength] * sizeof(unsigned char))/(1024 * 1024);
		printf("Total GPU space required: %lu\n", mem_req);

		// query device memory
		error = cudaMemGetInfo(&mem_free, &mem_total);
		printf("Total GPU memory: %lu\n", mem_total/(1024 * 1024));
		printf("Total GPU space available: %lu\n", mem_free/(1024 * 1024));
			
		if(mem_req < mem_free){
			unsigned char *d_byteCompressedDataOverflow;
			// malloc
			
			// allocate input file data
			error = cudaMalloc((void **)&d_inputFileData, inputFileLength * sizeof(unsigned char));
			if (error != cudaSuccess)
					printf("erro_1: %s\n", cudaGetErrorString(error));
				
			// allocate offset 
			error = cudaMalloc((void **)&d_compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int));
			if (error != cudaSuccess)
					printf("erro_2: %s\n", cudaGetErrorString(error));
				
			// allocate structure
			error = cudaMalloc((void **)&d_huffmanDictionary, sizeof(huffmanDictionary));
			if (error != cudaSuccess)
					printf("erro_3: %s\n", cudaGetErrorString(error));
				
			// allocate bitSequence to byte storage
			error = cudaMalloc((void **)&d_byteCompressedData, compressedDataOffset[integerOverflowIndex] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_5: %s\n", cudaGetErrorString(error));
				
			error = cudaMalloc((void **)&d_byteCompressedDataOverflow, compressedDataOffset[inputFileLength] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_6: %s\n", cudaGetErrorString(error));
							
			// memcpy
			// copy input data
			error = cudaMemcpy(d_inputFileData, inputFileData, inputFileLength * sizeof(unsigned char), cudaMemcpyHostToDevice);
			if (error!= cudaSuccess)
					printf("erro_7: %s\n", cudaGetErrorString(error));
				
			// copy offset
			error = cudaMemcpy(d_compressedDataOffset, compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
			if (error!= cudaSuccess)
					printf("erro_8: %s\n", cudaGetErrorString(error));
				
			// copy structure
			error = cudaMemcpy(d_huffmanDictionary, &huffmanDictionary, sizeof(huffmanDictionary), cudaMemcpyHostToDevice);
			if (error!= cudaSuccess)
					printf("erro_9: %s\n", cudaGetErrorString(error));
			
			// initialize d_byteCompressedData
			// initialize bitSequence to byte array to  0
			error = cudaMemset(d_byteCompressedData, 0, compressedDataOffset[integerOverflowIndex] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_10: %s\n", cudaGetErrorString(error));	
				
			error = cudaMemset(d_byteCompressedDataOverflow, 0, compressedDataOffset[inputFileLength] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_11: %s\n", cudaGetErrorString(error));		
				
			// copy constant memory
			if(constMemoryFlag == 1){
				error = cudaMemcpyToSymbol (d_bitSequenceConstMemory, bitSequenceConstMemory, 256 * 255 * sizeof(unsigned char));
				if (error!= cudaSuccess)
					printf("erro_12: %s\n", cudaGetErrorString(error));
			}
			
	
			// get GPU storage data after transfers
			error = cudaMemGetInfo(&mem_free, &mem_total);
			printf("Total GPU memory: %lu\n", mem_total/(1024 * 1024));
			printf("Total GPU space available: %lu\n", mem_free/(1024 * 1024));
			
			// launch kernel
			compress<<<1, 1024>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, d_byteCompressedDataOverflow, inputFileLength, constMemoryFlag, integerOverflowIndex);
			
			// check status
			cudaError_t error_kernel = cudaGetLastError();
			if (error_kernel != cudaSuccess)
				printf("erro_final: %s\n", cudaGetErrorString(error_kernel));
			
			// get output data
			if(bitPaddingFlag == 0){
				error = cudaMemcpy(inputFileData, d_inputFileData, (compressedDataOffset[integerOverflowIndex] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("erro_11: %s\n", cudaGetErrorString(error));
				error = cudaMemcpy(&inputFileData[(compressedDataOffset[integerOverflowIndex] / 8)], &d_inputFileData[(compressedDataOffset[integerOverflowIndex] / 8)], ((compressedDataOffset[inputFileLength] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("erro_12: %s\n", cudaGetErrorString(error));
			}
			else{
				error = cudaMemcpy(inputFileData, d_inputFileData, (compressedDataOffset[integerOverflowIndex] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("erro_11: %s\n", cudaGetErrorString(error));
				unsigned char temp_huffmanTreeNode = inputFileData[(compressedDataOffset[integerOverflowIndex] / 8) - 1];
				
				error = cudaMemcpy(&inputFileData[(compressedDataOffset[integerOverflowIndex] / 8) - 1], &d_inputFileData[(compressedDataOffset[integerOverflowIndex] / 8)], ((compressedDataOffset[inputFileLength] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("erro_12: %s\n", cudaGetErrorString(error));				
				inputFileData[(compressedDataOffset[integerOverflowIndex] / 8) - 1] = temp_huffmanTreeNode | inputFileData[(compressedDataOffset[integerOverflowIndex] / 8) - 1];
			}

			cudaFree(d_inputFileData);
			cudaFree(d_compressedDataOffset);
			cudaFree(d_huffmanDictionary);
			cudaFree(d_byteCompressedData);
			cudaFree(d_byteCompressedDataOverflow);
			
			// write src inputFileLength, header and compressed data to output file
			compressedFile = fopen(argv[2], "wb");
			fwrite(&inputFileLength, sizeof(unsigned int), 1, compressedFile);
			fwrite(frequency, sizeof(unsigned int), 256, compressedFile);
			fwrite(inputFileData, sizeof(unsigned char), (compressedDataOffset[inputFileLength] / 8 + compressedDataOffset[integerOverflowIndex] / 8) - 1, compressedFile);
			fclose(compressedFile);			
		}
	}
	// calculate run duration
	end = clock();
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("\ntime taken %d seconds and %d milliseconds\n\n", cpu_time_used / 1000, cpu_time_used % 1000);

	return 0;
}

// sortHuffmanTree nodes based on frequency
void sortHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes){
	int a, b;
	for (a = combinedHuffmanNodes; a < distinctCharacterCount - 1 + i; a++){
		for (b = combinedHuffmanNodes; b < distinctCharacterCount - 1 + i; b++){
			if (huffmanTreeNode[b].count > huffmanTreeNode[b + 1].count){
				temp_huffmanTreeNode = huffmanTreeNode[b];
				huffmanTreeNode[b] = huffmanTreeNode[b + 1];
				huffmanTreeNode[b + 1] = temp_huffmanTreeNode;
			}
		}
	}
}

// build tree based on sortHuffmanTree result
void buildHuffmanTree(int i, int distinctCharacterCount, int combinedHuffmanNodes){
	free(head_huffmanTreeNode);
	head_huffmanTreeNode = (struct huffmanTree *)malloc(sizeof(struct huffmanTree));
	head_huffmanTreeNode->count = huffmanTreeNode[combinedHuffmanNodes].count + huffmanTreeNode[combinedHuffmanNodes + 1].count;
	head_huffmanTreeNode->left = &huffmanTreeNode[combinedHuffmanNodes];
	head_huffmanTreeNode->right = &huffmanTreeNode[combinedHuffmanNodes + 1];
	huffmanTreeNode[distinctCharacterCount + i] = *head_huffmanTreeNode;
}

// get bitSequence sequence for each char value
void buildHuffmanDictionary(struct huffmanTree *root, unsigned char *bitSequence, unsigned char bitSequenceLength){
	if (root->left){
		bitSequence[bitSequenceLength] = 0;
		buildHuffmanDictionary(root->left, bitSequence, bitSequenceLength + 1);
	}

	if (root->right){
		bitSequence[bitSequenceLength] = 1;
		buildHuffmanDictionary(root->right, bitSequence, bitSequenceLength + 1);
	}

	if (root->left == NULL && root->right == NULL){
		huffmanDictionary.bitSequenceLength[root->letter] = bitSequenceLength;
		if(bitSequenceLength < 192){
			memcpy(huffmanDictionary.bitSequence[root->letter], bitSequence, bitSequenceLength * sizeof(unsigned char));
		}
		else{
			memcpy(bitSequenceConstMemory[root->letter], bitSequence, bitSequenceLength * sizeof(unsigned char));
			memcpy(huffmanDictionary.bitSequence[root->letter], bitSequence, 191);
			constMemoryFlag = 1;
		}
	}
}
