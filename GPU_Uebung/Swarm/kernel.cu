#include "kernel.cuh"
#include "cuda_device_array.h"

__global__ void add(int a, int b, int* sum)
{
	*sum = a + b;
}

__global__ void multiplyVM(double* matrix, double* vector, int cols, double* resultMatrix)
{
	//printf("block x: %d, block y: %d, thread x: %d\n", blockIdx.x, blockIdx.y, threadIdx.x);
	int const row = blockIdx.x;
	int const col = threadIdx.x;

	resultMatrix[(row * cols) + col] = matrix[(row * cols) + col] * vector[col];
}

__global__ void matrixVectorMultiplicationKernel(double* matrix, double* vector, double* result_matrix, int cols, int rows)
{
	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;
	if (ROW < rows && COL < cols) {
		result_matrix[(ROW * cols) + COL] = matrix[(ROW * cols) + COL] * vector[COL];
	} else result_matrix[ROW * cols + COL] = 0;
	//printf("result_matrix value: %f\n", result_matrix[(ROW * cols) + COL]);
}

__global__ void arrayAdd(double* matrix, int cols, int rows, double* result)
{
	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;
	
	//if (ROW < rows && COL < cols) {
		for (int i = 0; i < cols; i++)
		{
			result[ROW] += matrix[(i * cols) + COL];
		}
		//for (int i = 0; i < cols; i++)
		//{
		//	//printf("matrix value: %f\n", matrix[(ROW * cols) + COL]);
		//	result[ROW] += matrix[(ROW * cols) + COL];
		//}
	//}
	__syncthreads();
}

int add(int a, int b)
{
	int hostSum = 0;
	int* cudaSum = NULL;
	// Allocate Space
	CUDA_CHECK(cudaMalloc((void**)&cudaSum, sizeof(int)));

	add <<<1, 1>>> (a, b, cudaSum);
	CUDA_CHECK(cudaMemcpy(&hostSum, cudaSum, sizeof(int), cudaMemcpyDeviceToHost));

	// Free Space
	CUDA_CHECK(cudaFree(cudaSum));

	return hostSum;
}

void new_multiply()
{
	const int ROWS = 64;
	const int COLS = 100;

	const int SIZE = ROWS * COLS;

	// Declare on host
	double* host_A = new double[SIZE];
	for (int i = 0; i < SIZE; i++) host_A[i] = 1;

	double host_vector[COLS];
	for (int i = 0; i < COLS; i++) host_vector[i] = 1;

	double* host_result = new double[ROWS];

	// Allocate memory on the device
	DeviceArray<double> device_A(SIZE);
	DeviceArray<double> device_vector(COLS);
	DeviceArray<double> device_matrix_result(SIZE);
	DeviceArray<double> device_result(ROWS);

	device_A.set(&host_A[0], SIZE);
	device_vector.set(&host_vector[0], COLS);


	dim3 blocks_per_grid(1, 1);
	dim3 threads_per_block(COLS, ROWS);

	if (SIZE > 512)
	{
		threads_per_block.x = 32;
		threads_per_block.y = 32;
		blocks_per_grid.x = (unsigned int) ceil(double(COLS) / double(threads_per_block.x));
		blocks_per_grid.y = (unsigned int) ceil(double(ROWS) / double(threads_per_block.y));
	}
	std::cout << blocks_per_grid.x << std::endl;
	std::cout << blocks_per_grid.y << std::endl;
	matrixVectorMultiplicationKernel <<<blocks_per_grid, threads_per_block>>> (device_A.getData(), device_vector.getData(), device_matrix_result.getData(), COLS, ROWS);
	// multiplyVM <<<blocks_per_grid, threads_per_block>>>(device_A.getData(), device_vector.getData(), COLS, device_matrix_result.getData());
	device_matrix_result.get(&host_A[0], SIZE);
	//std::cout << "result:" << endl; // [20, 22]
	//for (int i = 0; i < ROWS; i++)
	//{
	//	for (int j = 0; j < COLS; j++)
	//	{
	//		std::cout << host_A[i * COLS + j] << " ";		
	//	}
	//	std::cout << std::endl;
	//}

	arrayAdd <<<blocks_per_grid, threads_per_block>>>(device_matrix_result.getData(), COLS, ROWS, device_result.getData());
	std::cout << device_result.getSize() << std::endl;
	device_result.get(&host_result[0], ROWS);
	cudaDeviceSynchronize();

	double* cpu_C;
	cpu_C = new double[ROWS];

	// Now do the matrix vector multiplication on the CPU
	double sum;
	for (int row = 0; row < ROWS; row++) {
		sum = 0.f;
		for (int col = 0; col < COLS; col++) {
			sum += host_A[(row * COLS) + col] * host_vector[col];
		}
		cpu_C[row] = sum;
	}

	double err = 0;
	// Check the result and make sure it is correct
	for (int ROW = 0; ROW < ROWS; ROW++) {
			err += cpu_C[ROW] - host_result[ROW];

	}

	cout << "Error: " << err << endl;

	//return 0;

	//std::cout << "result:" << endl; // [20, 22]
	//for (int i = 0; i < ROWS; i++)
	//{
	//	std::cout << (i+1) << ": \t" << host_result[i] << endl;
	//}
}

void multiply()
{
	const unsigned int rows = 100;
	const unsigned int cols = 100;

	// Declare on host
	double* A = new double[rows * cols];
	for (int i = 0; i < rows * cols; i++) A[i] = 1;

	double hostVector[cols];
	for (int i = 0; i < cols; i++) hostVector[i] = 1;
	double hostResult[rows];

	// Declare for CUDA
	double* cudaMatrix = NULL;
	double* cudaResultMatrix = NULL;
	double* cudaVector = NULL;
	double* cudaResult = NULL;

	// Allocate Space
	CUDA_CHECK(cudaMalloc((void**)&cudaMatrix, sizeof(double) * rows * cols));
	CUDA_CHECK(cudaMalloc((void**)&cudaResultMatrix, sizeof(double) * rows * cols));
	CUDA_CHECK(cudaMalloc((void**)&cudaVector, sizeof(double) * cols));
	CUDA_CHECK(cudaMalloc((void**)&cudaResult, sizeof(double) * rows));

	// Copy from Host to Device
	CUDA_CHECK(cudaMemcpy(cudaMatrix, A, sizeof(double) * rows * cols, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cudaVector, hostVector, sizeof(double) * cols, cudaMemcpyHostToDevice));

	// Run Calculation on Device
	//int blockCols = (int)ceil(cols / (double)BLOCK_WIDTH);
	//int blockRows = (int)ceil(rows / (double)BLOCK_HEIGHT);

	dim3 blockSettings(rows);
	dim3 threadSettings(cols);

	multiplyVM <<<blockSettings, threadSettings>>> (cudaMatrix, cudaVector, cols, cudaResultMatrix);

	//arrayAdd <<<blockSettings, 1>>> (cudaResultMatrix, cols, cudaResult);

	// Copy Result from Device to host
	CUDA_CHECK(cudaMemcpy(hostResult, cudaResult, sizeof(double) * rows, cudaMemcpyDeviceToHost));

	// Print out the Result
	std::cout << "result:" << endl; // [20, 22]
	for (int i = 0; i < rows; i++)
	{
		std::cout << hostResult[i] << endl;
	}

	// Free Space
	CUDA_CHECK(cudaFree(cudaVector));
	CUDA_CHECK(cudaFree(cudaMatrix));
	CUDA_CHECK(cudaFree(cudaResult));
}


	//glEnable( GL_DEPTH_TEST );

	//glGenVertexArrays( 1, &g_vertexArrayId );
	//glBindVertexArray( g_vertexArrayId );

	//GLuint vertexBufferId = 0;
	//glGenBuffers( 1, &vertexBufferId );
	//glBindBuffer( GL_ARRAY_BUFFER, vertexBufferId );
	//std::vector<GLfloat> const mesh = generateMesh();
	//glBufferData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * mesh.size(), &mesh[0], GL_STATIC_DRAW );

	//glEnableVertexAttribArray( 0 );
	//glBindBuffer( GL_ARRAY_BUFFER, vertexBufferId );
	//glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, ( void* ) NULL );

	//GLuint colorBufferId = 0;
	//glGenBuffers( 1, &colorBufferId );
	//glBindBuffer( GL_ARRAY_BUFFER, colorBufferId );
	//std::vector<GLfloat> const colors = generateColorData();
	//glBufferData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * colors.size(), &colors[0], GL_STATIC_DRAW );

	//glEnableVertexAttribArray( 1 );
	//glBindBuffer( GL_ARRAY_BUFFER, colorBufferId );
	//glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, 0, ( void* ) NULL );

	//glBindVertexArray( 0 );

	//g_shaderId = loadShaders( "vertex.glsl", "fragment.glsl" );

	
GLuint createVBO( std::vector<GLfloat> data )
{
	GLuint vbo;
	glGenBuffers( 1, &vbo );
	glBindBuffer( GL_ARRAY_BUFFER, vbo );
	glBufferData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * data.size(), &data[0], GL_STATIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	return vbo;
}