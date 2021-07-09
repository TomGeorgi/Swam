#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_device.h"
#include "kernel.cuh"

int main()
{
	CudaDevice device = CudaDevice(0);
	int a = 2, 
		b = 2;

	std::cout << device << std::endl;
	std::cout << "" << a << " + " << b << " = " << add(a, b) << std::endl;

	new_multiply();

	return 0;
}