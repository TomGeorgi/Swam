#include "kernel.cuh"

__global__ void add( int a , int b, int* sum )
{
	*sum = a + b;
}

int add(int a, int b)
{
	int hostSum = 0;
	int* cudaSum = NULL;
	// Allocate Space
	CUDA_CHECK( cudaMalloc((void**)&cudaSum, sizeof(int)) );

	add<<<1, 1>>>( a, b, cudaSum );
	CUDA_CHECK( cudaMemcpy( &hostSum, cudaSum, sizeof(int), cudaMemcpyDeviceToHost ) );

	// Free Space
	CUDA_CHECK( cudaFree( cudaSum ) );

	return hostSum;
}
