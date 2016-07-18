/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//GPU kernels
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

#include "parallelHeader.h"

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// single run and no overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
__global__ void compress(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, struct huffmanDictionary *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned int d_inputFileLength, unsigned int constMemoryFlag){
	__shared__ struct huffmanDictionary table;
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
}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// single run with overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
__global__ void compress(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, struct huffmanDictionary *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned char *d_temp_overflow, unsigned int d_inputFileLength, unsigned int constMemoryFlag, unsigned int overflowPosition){
	__shared__ struct huffmanDictionary table;
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
			memcpy(&d_temp_overflow[d_compressedDataOffset[(overflowPosition + 1)] - table.bitSequenceLength[d_inputFileData[overflowPosition]]], 
				   &table.bitSequence[d_inputFileData[overflowPosition]], table.bitSequenceLength[d_inputFileData[overflowPosition]]);
		}
	}
	// use constant memory and shared memory
	else{
		for(i = pos; i < overflowPosition; i += blockDim.x){
			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i]]; k++){
				if(k < 191)
					d_byteCompressedData[d_compressedDataOffset[i]+k] = table.bitSequence[d_inputFileData[i]][k];
				else
					d_byteCompressedData[d_compressedDataOffset[i]+k] = d_bitSequenceConstMemory[d_inputFileData[i]][k];
			}
		}
		for(i = overflowPosition + pos; i < inputFileLength - 1; i += blockDim.x){
			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i + 1]]; k++){
				if(k < 191)
					d_temp_overflow[d_compressedDataOffset[i + 1]+k] = table.bitSequence[d_inputFileData[i + 1]][k];
				else
					d_temp_overflow[d_compressedDataOffset[i + 1]+k] = d_bitSequenceConstMemory[d_inputFileData[i + 1]][k];
			}
		}
		if(pos == 0){
			memcpy(&d_temp_overflow[d_compressedDataOffset[(overflowPosition + 1)] - table.bitSequenceLength[d_inputFileData[overflowPosition]]], 
				   &d_bitSequenceConstMemory[d_inputFileData[overflowPosition]], table.bitSequenceLength[d_inputFileData[overflowPosition]]);
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

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// multiple run and no overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
__global__ void compress(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, struct huffmanDictionary *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned int d_lowerPosition, unsigned int constMemoryFlag, unsigned int d_upperPosition){
	__shared__ struct huffmanDictionary table;
	memcpy(&table, d_huffmanDictionary, sizeof(struct huffmanDictionary));
	//unsigned int inputFileLength = d_inputFileLength;
	unsigned int i, j, k;
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
	
	// when shared memory is sufficient
	if(constMemoryFlag == 0){
		for(i = pos + d_lowerPosition; i < d_upperPosition; i += blockDim.x){
			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i]]; k++){
				d_byteCompressedData[d_compressedDataOffset[i]+k] = table.bitSequence[d_inputFileData[i]][k];
			}
		}
		if(pos == 0 && d_lowerPosition != 0){
			memcpy(&d_byteCompressedData[d_compressedDataOffset[(d_lowerPosition)] - table.bitSequenceLength[d_inputFileData[d_lowerPosition - 1]]], 
				   &table.bitSequence[d_inputFileData[d_lowerPosition - 1]], table.bitSequenceLength[d_inputFileData[d_lowerPosition - 1]]);
		}
	}
	// use constant memory and shared memory
	else{
		for(i = pos + d_lowerPosition; i < d_upperPosition; i += blockDim.x){
			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i]]; k++){
				if(k < 191)
					d_byteCompressedData[d_compressedDataOffset[i]+k] = table.bitSequence[d_inputFileData[i]][k];
				else
					d_byteCompressedData[d_compressedDataOffset[i]+k] = d_bitSequenceConstMemory[d_inputFileData[i]][k];
			}
		}
		if(pos == 0 && d_lowerPosition != 0){
			memcpy(&d_byteCompressedData[d_compressedDataOffset[(d_lowerPosition)] - table.bitSequenceLength[d_inputFileData[d_lowerPosition - 1]]], 
				   &d_bitSequenceConstMemory[d_inputFileData[d_lowerPosition - 1]], table.bitSequenceLength[d_inputFileData[d_lowerPosition - 1]]);
		}
	}
	__syncthreads();
	
	for(i = pos * 8; i < d_compressedDataOffset[d_upperPosition]; i += blockDim.x * 8){
		for(j = 0; j < 8; j++){
			if(d_byteCompressedData[i + j] == 0){
				d_inputFileData[(i / 8)] = d_inputFileData[(i / 8)] << 1;
			}
			else{
				d_inputFileData[(i / 8)] = (d_inputFileData[i / 8] << 1) | 1;
			}
		}
	}
}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// multiple run and with overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
__global__ void compress(unsigned char *d_inputFileData, unsigned int *d_compressedDataOffset, struct huffmanDictionary *d_huffmanDictionary, unsigned char *d_byteCompressedData, unsigned char *d_temp_overflow, unsigned int d_lowerPosition, unsigned int constMemoryFlag, unsigned int d_upperPosition, unsigned int overflowPosition){
	__shared__ struct huffmanDictionary table;
	memcpy(&table, d_huffmanDictionary, sizeof(struct huffmanDictionary));
	//unsigned int inputFileLength = d_inputFileLength;
	unsigned int i, j, k;
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;	
	unsigned int offset_overflow;
	
	// when shared memory is sufficient
	if(constMemoryFlag == 0){
		for(i = pos + d_lowerPosition; i < overflowPosition; i += blockDim.x){
			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i]]; k++){
				d_byteCompressedData[d_compressedDataOffset[i]+k] = table.bitSequence[d_inputFileData[i]][k];
			}
		}
		for(i = overflowPosition + pos; i < d_upperPosition - 1; i += blockDim.x){
			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i + 1]]; k++){
				d_temp_overflow[d_compressedDataOffset[i + 1] + k] = table.bitSequence[d_inputFileData[i + 1]][k];
			}
		}
		if(pos == 0){
			memcpy(&d_temp_overflow[d_compressedDataOffset[(overflowPosition + 1)] - table.bitSequenceLength[d_inputFileData[overflowPosition]]], 
				   &table.bitSequence[d_inputFileData[overflowPosition]], table.bitSequenceLength[d_inputFileData[overflowPosition]]);
		}
		if(pos == 0 && d_lowerPosition != 0){
			memcpy(&d_byteCompressedData[d_compressedDataOffset[(d_lowerPosition)] - table.bitSequenceLength[d_inputFileData[d_lowerPosition - 1]]], 
				   &table.bitSequence[d_inputFileData[d_lowerPosition - 1]], table.bitSequenceLength[d_inputFileData[d_lowerPosition - 1]]);
		}
	}
	// use constant memory and shared memory
	else{
		for(i = pos + d_lowerPosition; i < d_upperPosition; i += blockDim.x){
			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i]]; k++){
				if(k < 191)
					d_byteCompressedData[d_compressedDataOffset[i]+k] = table.bitSequence[d_inputFileData[i]][k];
				else
					d_byteCompressedData[d_compressedDataOffset[i]+k] = d_bitSequenceConstMemory[d_inputFileData[i]][k];
			}
		}
		for(i = overflowPosition + pos; i < d_upperPosition - 1; i += blockDim.x){
			for(k = 0; k < table.bitSequenceLength[d_inputFileData[i + 1]]; k++){
				if(k < 191)
					d_byteCompressedData[d_compressedDataOffset[i]+k] = table.bitSequence[d_inputFileData[i]][k];
				else
					d_byteCompressedData[d_compressedDataOffset[i]+k] = d_bitSequenceConstMemory[d_inputFileData[i]][k];
			}
		}
		if(pos == 0){
			memcpy(&d_temp_overflow[d_compressedDataOffset[(overflowPosition + 1)] - table.bitSequenceLength[d_inputFileData[overflowPosition]]], 
				   &d_bitSequenceConstMemory[d_inputFileData[overflowPosition]], table.bitSequenceLength[d_inputFileData[overflowPosition]]);
		}
		if(pos == 0 && d_lowerPosition != 0){
			memcpy(&d_byteCompressedData[d_compressedDataOffset[(d_lowerPosition)] - table.bitSequenceLength[d_inputFileData[d_lowerPosition - 1]]], 
				   &d_bitSequenceConstMemory[d_inputFileData[d_lowerPosition - 1]], table.bitSequenceLength[d_inputFileData[d_lowerPosition - 1]]);
		}
	}
	__syncthreads();
	
	for(i = pos * 8; i < d_compressedDataOffset[overflowPosition]; i += blockDim.x * 8){
		for(j = 0; j < 8; j++){
			if(d_byteCompressedData[i + j] == 0){
				d_inputFileData[(i / 8)] = d_inputFileData[(i / 8)] << 1;
			}
			else{
				d_inputFileData[(i / 8)] = (d_inputFileData[i / 8] << 1) | 1;
			}
		}
	}

	offset_overflow = d_compressedDataOffset[overflowPosition] / 8;
	
	for(i = pos * 8; i < d_compressedDataOffset[d_upperPosition]; i += blockDim.x * 8){
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
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*---------------------------------------------------------------------------------------------------------------------------------------------*/