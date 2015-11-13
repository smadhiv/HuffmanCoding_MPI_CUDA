#include "header_cuda.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

__global__ void compress(unsigned char *input, unsigned int *offset, struct table *table, unsigned char *temp, unsigned int nints)
{
	__shared__ struct table d_table[256];

	unsigned int i, j, k;
	unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(pos == 0);
		memcpy(d_table, table, 256*sizeof(struct table));
	__syncthreads();
	
	for(i = pos; i < nints; i += blockDim.x)
	{
		for(k = 0; k < d_table[input[i]].size; k++)
		{
			temp[offset[i]+k] = d_table[input[i]].bit[k];
		}
	}
	__syncthreads();
	
	for(i = pos * 8; i < offset[nints]; i += blockDim.x * 8)
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

extern "C" void gpuCompress(unsigned int nints, unsigned char *h_input, unsigned int *h_offset, struct table *h_table)
{
	unsigned char *d_input, *d_temp;
	unsigned int *d_offset;
	struct table *d_table;
	cudaError_t error;

	error = cudaMalloc((void **)&d_input, nints*sizeof(unsigned char));
	if (error != cudaSuccess)
			printf("erro_1: %s\n", cudaGetErrorString(error));
		
	error = cudaMalloc((void **)&d_offset, (nints + 1)*sizeof(unsigned int));
	if (error != cudaSuccess)
			printf("erro_3: %s\n", cudaGetErrorString(error));
		
		
	error = cudaMalloc((void **)&d_table, 256*sizeof(table));
	if (error != cudaSuccess)
			printf("erro_4: %s\n", cudaGetErrorString(error));
		
		
	error = cudaMalloc((void **)&d_temp, h_offset[nints]*sizeof(unsigned char));
	cudaMemset(d_temp, 0, h_offset[nints]*sizeof(unsigned char));
	if (error!= cudaSuccess)
			printf("erro_5: %s\n", cudaGetErrorString(error));
	
	
	printf("Total GPU space: %.3fMB\n", (nints*sizeof(unsigned char) +
										//(h_offset[nints]/8)*sizeof(unsigned char) +
										(nints + 1)*sizeof(unsigned int) +
										256*sizeof(table) +
										h_offset[nints]*sizeof(unsigned char))/1000000.0);
	
	error = cudaMemcpy(d_input, h_input, nints*sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
				printf("erro_6: %s\n", cudaGetErrorString(error));

			
	error = cudaMemcpy(d_offset, h_offset, (nints + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
				printf("erro_7: %s\n", cudaGetErrorString(error));
			
			
	error = cudaMemcpy(d_table, h_table, 256 * sizeof(table), cudaMemcpyHostToDevice);
	if (error!= cudaSuccess)
				printf("erro_8: %s\n", cudaGetErrorString(error));

	compress<<<1, 1024>>>(d_input, d_offset, d_table, d_temp, nints);
	
	cudaMemcpy(h_input, d_input, ((h_offset[nints]/8))*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	//DEBUG PRINT TABLE
	/*struct table *out_table = (struct table *)malloc(256*sizeof(struct table));
	cudaMemcpy(out_table, d_table, 256*sizeof(table), cudaMemcpyDeviceToHost);
	for(int i = 0; i < 256; i++)
		printf("%d\t%d\t\n", h_table[i].size, out_table[i].size);
	free(out_table);*/
	
	cudaError_t error_final = cudaGetLastError();
	if (error_final != cudaSuccess)
		printf("erro_final: %s\n", cudaGetErrorString(error_final));
	
	cudaFree(d_input);
	cudaFree(d_offset);
	cudaFree(d_table);
	cudaFree(d_temp);
}

