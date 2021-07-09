#pragma once
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "macros.h"

using namespace std;


class CudaDevice
{
private:

	cudaDeviceProp properties;
	int deviceIndex;

public:

	CudaDevice();
	CudaDevice(cudaDeviceProp properties, int deviceIndex);
	CudaDevice(int deviceIndex);

	CudaDevice(const CudaDevice& cdv);
	CudaDevice& operator=(const CudaDevice& cdv);

	~CudaDevice() {};

	friend ostream& operator<<(ostream& os, const CudaDevice& dv);
};



