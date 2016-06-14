//Sriram Madhivanan
//Struct of Arrays
//Constant memory if dictinary goes beyond 191 bits
//Max possible shared memory
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../Huffman/huffman.h"

#define block_size 1024
unsigned char __constant__ d_bitSequenceConstMemory[256][255];

__global__ void compress(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, struct huffmanDict *d_huffmanDictionary, 
						 unsigned char *d_byteCompressedData, unsigned int d_inputFileLength, unsigned int constMemoryFlag);

__global__ void compress(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, struct huffmanDict *d_huffmanDictionary, 
						 unsigned char *d_byteCompressedData, unsigned char *d_temp_overflow, unsigned int d_inputFileLength, unsigned int constMemoryFlag, 
						 unsigned int overflowPosition);

extern "C" int wrapperGPU(char **file, unsigned char *inputFileData, int inputFileLength){
	unsigned int i;
	unsigned int frequency[256];
	unsigned char *d_inputFileData, *d_byteCompressedData;
	unsigned int *d_compressedDataOffset, *compressedDataOffset;
	struct huffmanDict *d_huffmanDictionary;
	unsigned int integerOverflowFlag, integerOverflowIndex, bitPaddingFlag;
	FILE *compressedFile;
	cudaError_t error;

	printf("%d\n", inputFileLength);
	// calculate compressed data offset - (1048576 is a safe number that will ensure there is no integer overflow in GPU, it should be minimum 8 * number of threads)
	integerOverflowFlag = 0;
	bitPaddingFlag = 0;
	compressedDataOffset = (unsigned int *)malloc((inputFileLength + 1) * sizeof(unsigned int));
	compressedDataOffset[0] = 0;
	for(i = 0; i < inputFileLength; i++){
		compressedDataOffset[i + 1] = huffmanDictionary.bitSequenceLength[inputFileData[i]] + compressedDataOffset[i];
		if(compressedDataOffset[i + 1] + 1048576 < compressedDataOffset[i]){
			printf("Overflow error occured\n");
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

	printf("doing stuff 2\n");

	if(integerOverflowFlag == 0){
		long unsigned int mem_free, mem_total;
		long unsigned int mem_req;
		mem_req = 2 + (inputFileLength * sizeof(unsigned char) 
			+ (inputFileLength + 1) * sizeof(unsigned int) 
			+ sizeof(huffmanDict) 
			+ (long unsigned int)compressedDataOffset[inputFileLength] * sizeof(unsigned char))
			/(1024 * 1024);
		
		// query device memory
		error = cudaMemGetInfo(&mem_free, &mem_total);
		printf("Total GPU memory: %lu\n", mem_total/(1024 * 1024));
		printf("Total GPU space available: %lu\n", mem_free/(1024 * 1024));
		printf("Total GPU space required: %lu\n", mem_req);

		if(mem_req < mem_free){
			// malloc
			error = cudaMalloc((void **)&d_inputFileData, inputFileLength * sizeof(unsigned char));
			if (error != cudaSuccess)
					printf("erro_1: %s\n", cudaGetErrorString(error));
			error = cudaMalloc((void **)&d_compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int));
			if (error != cudaSuccess)
					printf("erro_2: %s\n", cudaGetErrorString(error));
			error = cudaMalloc((void **)&d_huffmanDictionary, sizeof(huffmanDict));
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
			error = cudaMemcpy(d_huffmanDictionary, &huffmanDictionary, sizeof(huffmanDict), cudaMemcpyHostToDevice);
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
			printf("Total GPU space left: %lu\n", mem_free/(1024 * 1024));
		
			compress<<<1, block_size>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, inputFileLength, constMemoryFlag);
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
			compressedFile = fopen(*file, "wb");
			fwrite(&inputFileLength, sizeof(unsigned int), 1, compressedFile);
			fwrite(frequency, sizeof(unsigned int), 256, compressedFile);
			fwrite(inputFileData, sizeof(unsigned char), (compressedDataOffset[inputFileLength] / 8), compressedFile);
			fclose(compressedFile);			
		}
	}
	else{
		long unsigned int mem_free, mem_total;
		long unsigned int mem_req;
		mem_req = 2 + (long unsigned int)((long unsigned int)inputFileLength * sizeof(unsigned char) 
					+ (long unsigned int)(inputFileLength + 1) * sizeof(unsigned int) 
					+ sizeof(huffmanDict) 
					+ (long unsigned int)compressedDataOffset[integerOverflowIndex] * sizeof(unsigned char) 
					+ (long unsigned int)compressedDataOffset[inputFileLength] * sizeof(unsigned char))
					/(1024 * 1024);
		mem_req = mem_req * (1024 * 1024);

		// query device memory
		error = cudaMemGetInfo(&mem_free, &mem_total);
		printf("Total GPU memory: %lu\n", mem_total/(1024 * 1024));
		printf("Total GPU space available: %lu\n", mem_free/(1024 * 1024));
		printf("Total GPU space required: %lu\n", mem_req/(1024 * 1024));

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
			error = cudaMalloc((void **)&d_huffmanDictionary, sizeof(huffmanDict));
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
			error = cudaMemcpy(d_huffmanDictionary, &huffmanDictionary, sizeof(huffmanDict), cudaMemcpyHostToDevice);
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
		
			// run kernel and copy output
			error = cudaMemGetInfo(&mem_free, &mem_total);
			printf("Total GPU space left: %lu\n", mem_free/(1024 * 1024));
			
			// launch kernel
			compress<<<1, block_size>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, d_byteCompressedDataOverflow, inputFileLength, constMemoryFlag, integerOverflowIndex);
			
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
			compressedFile = fopen(*file, "wb");
			fwrite(&inputFileLength, sizeof(unsigned int), 1, compressedFile);
			fwrite(frequency, sizeof(unsigned int), 256, compressedFile);
			fwrite(inputFileData, sizeof(unsigned char), (compressedDataOffset[inputFileLength] / 8 + compressedDataOffset[integerOverflowIndex] / 8) - 1, compressedFile);
			fclose(compressedFile);			
		}
	}
	return 0;
}