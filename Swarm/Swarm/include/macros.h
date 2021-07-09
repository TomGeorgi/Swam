#pragma once
#include <iostream>
#include <stdexcept>
#include <functional>

#include "cuda_runtime.h"

using namespace std;

#define CUDA_CHECK( func ) \
{ \
	cudaError_t error_code = func; \
	if (error_code != cudaSuccess) \
		cerr << cudaGetErrorName(error_code) << ": " << cudaGetErrorString(error_code) << endl; \
}

