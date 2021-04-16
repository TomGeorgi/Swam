#include "kernel.cuh"

__global__ void add( int a , int b, int* sum )
{
	*sum = a + b;
}

__global__ void multiplyVM(double* matrix, double* vector, int rows, int cols, double* result)
{
	printf("rows %i, cols %i\n", rows, cols);
	for (int c = 0; c < cols; c++) printf("%d\n", vector[c]);
	for (int r = 0; r < rows; r++)
	{
		double tmp = 0;
		for (int c = 0; c < cols; c++)
		{
			double m_val = matrix[(r * cols) + c];
			double v_val = vector[c];
			printf("m_val = %d\n", m_val);
			printf("v_val = %d\n", v_val);
			printf("r = %i, c = %i\n", r, c);
			tmp += m_val * v_val;
			printf("tmp = %d\n", tmp);
		}
		result[r] = tmp;
	}

	printf("multiplying finis\n");
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

void multiply()
{
	const int rows = 2;
	const int cols = 3;

	// Declare on host
	double** A = new double* [rows];
	A[0] = new double[rows * cols]{2, 5, 3, 3, 4, 2};
	for (int i = 1; i < rows; ++i) A[i] = A[i - 1] + cols;

	double hostVector[cols] = { 2, 3, 1 };
	double hostResult[cols] = { 0, 0 };

	// Declare for CUDA
	double* cudaMatrix = NULL;
	double* cudaVector = NULL;
	double* cudaResult = NULL;

	// Allocate Space
	std::cout << "allocate memory" << std::endl;
	CUDA_CHECK(cudaMalloc((void**)&cudaMatrix, sizeof(double) * rows * cols));
	CUDA_CHECK(cudaMalloc((void**)&cudaVector, sizeof(double) * cols));
	CUDA_CHECK(cudaMalloc((void**)&cudaResult, sizeof(double) * rows));

	std::cout << "memcpy to dev" << std::endl;
	CUDA_CHECK(cudaMemcpy(cudaMatrix, A[0], sizeof(double) * rows * cols, cudaMemcpyHostToDevice));
	std::cout << "memcpy to dev" << std::endl;
	CUDA_CHECK(cudaMemcpy(cudaVector, hostVector, sizeof(double) * cols, cudaMemcpyHostToDevice));

	multiplyVM<<<1, 1>>>(cudaMatrix, cudaVector, rows, cols, cudaResult);
	CUDA_CHECK(cudaMemcpy(hostResult, cudaResult, sizeof(double) * rows, cudaMemcpyDeviceToHost));

	cout << "result:" << endl; // [20, 20]
	for (int i = 0; i < rows; i++)
	{
		cout << hostResult[i] << endl;
	}
	CUDA_CHECK(cudaFree(cudaVector));
	CUDA_CHECK(cudaFree(cudaMatrix));
	CUDA_CHECK(cudaFree(cudaResult));
}
