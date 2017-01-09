/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//GPU wrapper for GPU/MPI-CUDA Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/parallelHeader.h"
#define block_size 1024
__constant__ unsigned char d_bitSequenceConstMemory[256][255];

void lauchCUDAHuffmanCompress(unsigned char *inputFileData, unsigned int *compressedDataOffset, unsigned int inputFileLength, int numKernelRuns, unsigned int integerOverflowFlag, long unsigned int mem_req){
	int i;
	unsigned char *d_inputFileData, *d_byteCompressedData;
	unsigned int *d_compressedDataOffset;
	struct huffmanDictionary *d_huffmanDictionary;
	unsigned int *gpuBitPaddingFlag, *bitPaddingFlag;
	unsigned int *gpuMemoryOverflowIndex, *integerOverflowIndex;
	long unsigned int mem_free, mem_total;
	cudaError_t error;
	
	// generate offset 
	if(integerOverflowFlag == 0){
		// only one time run of kernel
		if(numKernelRuns == 1){
			createDataOffsetArray(compressedDataOffset, inputFileData, inputFileLength);
		}
		
		// multiple run of kernel due to larger file or smaller gpu memory
		else{
			gpuBitPaddingFlag = (unsigned int *)calloc(numKernelRuns, sizeof(unsigned int));
			gpuMemoryOverflowIndex = (unsigned int *)calloc(numKernelRuns * 2, sizeof(unsigned int));
			createDataOffsetArray(compressedDataOffset, inputFileData, inputFileLength, gpuMemoryOverflowIndex, gpuBitPaddingFlag, mem_req);
		}
	}
	
	// when there is integer over flow
	else{
		// overflow occurs and single run
		if(numKernelRuns == 1){
			bitPaddingFlag = (unsigned int *)calloc(numKernelRuns, sizeof(unsigned int));
			integerOverflowIndex = (unsigned int *)calloc(numKernelRuns * 2, sizeof(unsigned int));
			createDataOffsetArray(compressedDataOffset, inputFileData, inputFileLength, integerOverflowIndex, bitPaddingFlag, 10240);
		}
		
		// overflow occurs and multiple run
		else{
			gpuBitPaddingFlag = (unsigned int *)calloc(numKernelRuns, sizeof(unsigned int));
			bitPaddingFlag = (unsigned int *)calloc(numKernelRuns, sizeof(unsigned int));
			integerOverflowIndex = (unsigned int *)calloc(numKernelRuns * 2, sizeof(unsigned int));
			gpuMemoryOverflowIndex = (unsigned int *)calloc(numKernelRuns * 2, sizeof(unsigned int));
			createDataOffsetArray(compressedDataOffset, inputFileData, inputFileLength, integerOverflowIndex, bitPaddingFlag, gpuMemoryOverflowIndex, gpuBitPaddingFlag, 10240, mem_req);	
		}
	}
	
	// GPU initiation
	{	
		// allocate memory for input data, offset information and dictionary
		error = cudaMalloc((void **)&d_inputFileData, inputFileLength * sizeof(unsigned char));
		if (error != cudaSuccess)
				printf("erro_1: %s\n", cudaGetErrorString(error));
			
		error = cudaMalloc((void **)&d_compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int));
		if (error != cudaSuccess)
				printf("erro_2: %s\n", cudaGetErrorString(error));
		error = cudaMalloc((void **)&d_huffmanDictionary, sizeof(huffmanDictionary));
		if (error != cudaSuccess)
				printf("erro_3: %s\n", cudaGetErrorString(error));
			
		// memory copy input data, offset information and dictionary
		error = cudaMemcpy(d_inputFileData, inputFileData, inputFileLength * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (error!= cudaSuccess)
				printf("erro_4: %s\n", cudaGetErrorString(error));
		error = cudaMemcpy(d_compressedDataOffset, compressedDataOffset, (inputFileLength + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
		if (error!= cudaSuccess)
				printf("erro_5: %s\n", cudaGetErrorString(error));
		error = cudaMemcpy(d_huffmanDictionary, &huffmanDictionary, sizeof(huffmanDictionary), cudaMemcpyHostToDevice);
		if (error!= cudaSuccess)
				printf("erro_6: %s\n", cudaGetErrorString(error));
			
		// copy constant memory if required for dictionary
		if(constMemoryFlag == 1){
			error = cudaMemcpyToSymbol (d_bitSequenceConstMemory, bitSequenceConstMemory, 256 * 255 * sizeof(unsigned char));
			if (error!= cudaSuccess)
				printf("erro_const: %s\n", cudaGetErrorString(error));
		}
	}

	
	// memory copy of offset data
	if(numKernelRuns == 1){
		// no overflow
		if(integerOverflowFlag == 0){
			error = cudaMalloc((void **)&d_byteCompressedData, (compressedDataOffset[inputFileLength]) * sizeof(unsigned char));
			if (error!= cudaSuccess)
				printf("erro_7: %s\n", cudaGetErrorString(error));
			
			// initialize d_byteCompressedData 
			error = cudaMemset(d_byteCompressedData, 0, compressedDataOffset[inputFileLength] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_8: %s\n", cudaGetErrorString(error));
			
			// debug
			if(1){
				cudaMemGetInfo(&mem_free, &mem_total);
				printf("Free Mem: %lu\n", mem_free);		
			}			
			
			// run kernel
			compress<<<1, block_size>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, inputFileLength, constMemoryFlag);
			cudaError_t error_kernel = cudaGetLastError();
			if (error_kernel != cudaSuccess)
				printf("erro_final: %s\n", cudaGetErrorString(error_kernel));
			
			// copy compressed data from GPU to CPU memory
			error = cudaMemcpy(inputFileData, d_inputFileData, ((compressedDataOffset[inputFileLength] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			if (error != cudaSuccess)
				printf("erro_9: %s\n", cudaGetErrorString(error));
			
			// free allocated memory
			cudaFree(d_inputFileData);
			cudaFree(d_compressedDataOffset);
			cudaFree(d_huffmanDictionary);
			cudaFree(d_byteCompressedData);
		}
		
		// integer overflow occurs
		else{
			// additional variable to store offset data after integer oveflow
			unsigned char *d_byteCompressedDataOverflow;
			
			// allocate memory to store offset information
			error = cudaMalloc((void **)&d_byteCompressedData, compressedDataOffset[integerOverflowIndex[0]] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_7: %s\n", cudaGetErrorString(error));
			error = cudaMalloc((void **)&d_byteCompressedDataOverflow, compressedDataOffset[inputFileLength] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_8: %s\n", cudaGetErrorString(error));
			
			// initialize offset data
			error = cudaMemset(d_byteCompressedData, 0, compressedDataOffset[integerOverflowIndex[0]] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_9: %s\n", cudaGetErrorString(error));	
			error = cudaMemset(d_byteCompressedDataOverflow, 0, compressedDataOffset[inputFileLength] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_10: %s\n", cudaGetErrorString(error));
			
			// debug
			if(1){
				cudaMemGetInfo(&mem_free, &mem_total);
				printf("Free Mem: %lu\n", mem_free);		
			}
			
			// launch kernel
			compress<<<1, block_size>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, d_byteCompressedDataOverflow, inputFileLength, constMemoryFlag, integerOverflowIndex[0]);
			
			// check status
			cudaError_t error_kernel = cudaGetLastError();
			if (error_kernel != cudaSuccess)
				printf("erro_final: %s\n", cudaGetErrorString(error_kernel));
			
			// get output data
			if(bitPaddingFlag[0] == 0){
				error = cudaMemcpy(inputFileData, d_inputFileData, (compressedDataOffset[integerOverflowIndex[0]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("erro_11: %s\n", cudaGetErrorString(error));
				error = cudaMemcpy(&inputFileData[(compressedDataOffset[integerOverflowIndex[0]] / 8)], &d_inputFileData[(compressedDataOffset[integerOverflowIndex[0]] / 8)], ((compressedDataOffset[inputFileLength] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("erro_12: %s\n", cudaGetErrorString(error));
			}
			else{
				error = cudaMemcpy(inputFileData, d_inputFileData, (compressedDataOffset[integerOverflowIndex[0]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("erro_11: %s\n", cudaGetErrorString(error));
				unsigned char temp_compByte = inputFileData[(compressedDataOffset[integerOverflowIndex[0]] / 8) - 1];
				
				error = cudaMemcpy(&inputFileData[(compressedDataOffset[integerOverflowIndex[0]] / 8) - 1], &d_inputFileData[(compressedDataOffset[integerOverflowIndex[0]] / 8)], ((compressedDataOffset[inputFileLength] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				if (error != cudaSuccess)
					printf("erro_12: %s\n", cudaGetErrorString(error));				
				inputFileData[(compressedDataOffset[integerOverflowIndex[0]] / 8) - 1] = temp_compByte | inputFileData[(compressedDataOffset[integerOverflowIndex[0]] / 8) - 1];
			}

			// free allocated memory
			cudaFree(d_inputFileData);
			cudaFree(d_compressedDataOffset);
			cudaFree(d_huffmanDictionary);
			cudaFree(d_byteCompressedData);
			cudaFree(d_byteCompressedDataOverflow);
		}
	}
	
	else{
		if(integerOverflowFlag == 0){
			error = cudaMalloc((void **)&d_byteCompressedData, (compressedDataOffset[gpuMemoryOverflowIndex[1]]) * sizeof(unsigned char));
			if (error!= cudaSuccess)
				printf("erro_7: %s\n", cudaGetErrorString(error));

			// debug
			if(1){
				cudaMemGetInfo(&mem_free, &mem_total);
				printf("Free Mem: %lu\n", mem_free);		
			}		
			
			unsigned int pos = 0;
			for(i = 0; i < numKernelRuns; i++){
				// initialize d_byteCompressedData 
				error = cudaMemset(d_byteCompressedData, 0, compressedDataOffset[gpuMemoryOverflowIndex[1]] * sizeof(unsigned char));
				if (error!= cudaSuccess)
						printf("erro_8: %s\n", cudaGetErrorString(error));
	
				compress<<<1, block_size>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, gpuMemoryOverflowIndex[i * 2], constMemoryFlag, gpuMemoryOverflowIndex[i * 2 + 1]);
				cudaError_t error_kernel = cudaGetLastError();
				if (error_kernel != cudaSuccess)
					printf("erro_final: %s\n", cudaGetErrorString(error_kernel));	
	
	
				if(gpuBitPaddingFlag[i] == 0){
					error = cudaMemcpy(&inputFileData[pos], d_inputFileData, (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
					if (error != cudaSuccess)
						printf("erro_12: %s\n", cudaGetErrorString(error));
					pos += (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8);
				}
				else{
					unsigned char temp_compByte = inputFileData[pos - 1];
					error = cudaMemcpy(&inputFileData[pos - 1], d_inputFileData, ((compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
					if (error != cudaSuccess)
						printf("erro_12: %s\n", cudaGetErrorString(error));
					inputFileData[pos - 1] = temp_compByte | inputFileData[pos - 1];
					pos +=  (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8) - 1;
				}
			}
	
			
			// free allocated memory
			cudaFree(d_inputFileData);
			cudaFree(d_compressedDataOffset);
			cudaFree(d_huffmanDictionary);
			cudaFree(d_byteCompressedData);
		}
		
		else{
			// additional variable to store offset data after integer oveflow
			unsigned char *d_byteCompressedDataOverflow;
			error = cudaMalloc((void **)&d_byteCompressedData, (compressedDataOffset[integerOverflowIndex[0]]) * sizeof(unsigned char));
			if (error!= cudaSuccess)
				printf("erro_7: %s\n", cudaGetErrorString(error));
			error = cudaMalloc((void **)&d_byteCompressedDataOverflow, compressedDataOffset[gpuMemoryOverflowIndex[1]] * sizeof(unsigned char));
			if (error!= cudaSuccess)
					printf("erro_8: %s\n", cudaGetErrorString(error));

			// debug
			if(1){
				cudaMemGetInfo(&mem_free, &mem_total);
				printf("Free Mem: %lu\n", mem_free);		
			}		
			
			unsigned int pos = 0;
			for(i = 0; i < numKernelRuns; i++){
				if(integerOverflowIndex[i] != 0){
					// initialize d_byteCompressedData 
					error = cudaMemset(d_byteCompressedData, 0, compressedDataOffset[integerOverflowIndex[0]] * sizeof(unsigned char));
					if (error!= cudaSuccess)
							printf("erro_9: %s\n", cudaGetErrorString(error));
					error = cudaMemset(d_byteCompressedDataOverflow, 0, compressedDataOffset[gpuMemoryOverflowIndex[1]] * sizeof(unsigned char));
					if (error!= cudaSuccess)
							printf("erro_10: %s\n", cudaGetErrorString(error));
		
					compress<<<1, block_size>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, d_byteCompressedDataOverflow, gpuMemoryOverflowIndex[i * 2], constMemoryFlag, gpuMemoryOverflowIndex[i * 2 + 1], integerOverflowIndex[i]);
					cudaError_t error_kernel = cudaGetLastError();
					if (error_kernel != cudaSuccess)
						printf("erro_final: %s\n", cudaGetErrorString(error_kernel));	
		
					if(gpuBitPaddingFlag[i] == 0){
						if(bitPaddingFlag[i] == 0){
							error = cudaMemcpy(&inputFileData[pos], d_inputFileData, (compressedDataOffset[integerOverflowIndex[i]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
								printf("erro_11: %s\n", cudaGetErrorString(error));
							error = cudaMemcpy(&inputFileData[pos + (compressedDataOffset[integerOverflowIndex[i]] / 8)], &d_inputFileData[(compressedDataOffset[integerOverflowIndex[i]] / 8)], ((compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
								printf("erro_12: %s\n", cudaGetErrorString(error));
							pos += (compressedDataOffset[integerOverflowIndex[i]] / 8) + (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8);
						}
						else{
							error = cudaMemcpy(&inputFileData[pos], d_inputFileData, (compressedDataOffset[integerOverflowIndex[i]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
								printf("erro_11: %s\n", cudaGetErrorString(error));
							unsigned char temp_compByte = inputFileData[pos + (compressedDataOffset[integerOverflowIndex[i]] / 8) - 1];
							
							error = cudaMemcpy(&inputFileData[pos + (compressedDataOffset[integerOverflowIndex[i]] / 8) - 1], &d_inputFileData[(compressedDataOffset[integerOverflowIndex[i]] / 8)], ((compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
								printf("erro_12: %s\n", cudaGetErrorString(error));				
							inputFileData[pos + (compressedDataOffset[integerOverflowIndex[i]] / 8) - 1] = temp_compByte | inputFileData[pos + (compressedDataOffset[integerOverflowIndex[i]] / 8) - 1];
							pos += (compressedDataOffset[integerOverflowIndex[i]] / 8) + (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8) - 1;
						}
					}
					else{
						unsigned char temp_gpuCompByte = inputFileData[pos - 1];
						if(bitPaddingFlag[i] == 0){
							error = cudaMemcpy(&inputFileData[pos - 1], d_inputFileData, (compressedDataOffset[integerOverflowIndex[i]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
								printf("erro_11: %s\n", cudaGetErrorString(error));
							error = cudaMemcpy(&inputFileData[pos -1 + (compressedDataOffset[integerOverflowIndex[i]] / 8)], &d_inputFileData[(compressedDataOffset[integerOverflowIndex[i]] / 8)], ((compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
								printf("erro_12: %s\n", cudaGetErrorString(error));
							inputFileData[pos - 1] = temp_gpuCompByte | inputFileData[pos - 1];
							pos += (compressedDataOffset[integerOverflowIndex[i]] / 8) + (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8) - 1;
						}
						else{
							error = cudaMemcpy(&inputFileData[pos - 1], d_inputFileData, (compressedDataOffset[integerOverflowIndex[i]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
								printf("erro_11: %s\n", cudaGetErrorString(error));
							unsigned char temp_compByte = inputFileData[ pos -1 + (compressedDataOffset[integerOverflowIndex[i]] / 8) - 1];
							
							error = cudaMemcpy(&inputFileData[(compressedDataOffset[integerOverflowIndex[i]] / 8) - 1], &d_inputFileData[(compressedDataOffset[integerOverflowIndex[i]] / 8)], ((compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
							if (error != cudaSuccess)
								printf("erro_12: %s\n", cudaGetErrorString(error));				
							inputFileData[(compressedDataOffset[pos - 1 + integerOverflowIndex[i]] / 8) - 1] = temp_compByte | inputFileData[pos - 1 + (compressedDataOffset[integerOverflowIndex[i]] / 8) - 1];
							inputFileData[pos - 1] = temp_gpuCompByte | inputFileData[pos - 1];
							pos += (compressedDataOffset[integerOverflowIndex[i]] / 8) + (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8) - 2;
						}
					}			
				}
				else{
					// initialize d_byteCompressedData
					error = cudaMemset(d_byteCompressedData, 0, compressedDataOffset[integerOverflowIndex[0]] * sizeof(unsigned char));
					if (error!= cudaSuccess)
							printf("erro_8: %s\n", cudaGetErrorString(error));
		
					compress<<<1, block_size>>>(d_inputFileData, d_compressedDataOffset, d_huffmanDictionary, d_byteCompressedData, gpuMemoryOverflowIndex[i * 2], constMemoryFlag, gpuMemoryOverflowIndex[i * 2 + 1]);
					cudaError_t error_kernel = cudaGetLastError();
					if (error_kernel != cudaSuccess)
						printf("erro_final: %s\n", cudaGetErrorString(error_kernel));	
		
		
					if(gpuBitPaddingFlag[i] == 0){
						error = cudaMemcpy(&inputFileData[pos], d_inputFileData, (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
						if (error != cudaSuccess)
							printf("erro_12: %s\n", cudaGetErrorString(error));
						pos += (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8);
					}
					else{
						unsigned char temp_huffmanTreeNode = inputFileData[pos - 1];
						error = cudaMemcpy(&inputFileData[pos - 1], d_inputFileData, ((compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8)) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
						if (error != cudaSuccess)
							printf("erro_12: %s\n", cudaGetErrorString(error));
						inputFileData[pos - 1] = temp_huffmanTreeNode | inputFileData[pos - 1];
						pos +=  (compressedDataOffset[gpuMemoryOverflowIndex[i * 2 + 1]] / 8) - 1;
					}			
				}
			}
	
			// free allocated memory
			cudaFree(d_inputFileData);
			cudaFree(d_compressedDataOffset);
			cudaFree(d_huffmanDictionary);
			cudaFree(d_byteCompressedData);		
		}
	}
}