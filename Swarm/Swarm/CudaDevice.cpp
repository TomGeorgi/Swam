#include "CudaDevice.h"

CudaDevice::CudaDevice()
{
	int deviceCounter;
	CUDA_CHECK( cudaGetDeviceCount(&deviceCounter) );
	if (deviceCounter != 0)
	{
		deviceIndex = 0;
		CUDA_CHECK(cudaGetDeviceProperties(&properties, deviceIndex));
		CUDA_CHECK(cudaSetDevice(deviceIndex));
	}
}

CudaDevice::CudaDevice(cudaDeviceProp properties, int deviceIndex): properties(properties), 
																	deviceIndex(deviceIndex)
{
	CUDA_CHECK(cudaGetDeviceProperties(&properties, deviceIndex));
	CUDA_CHECK(cudaSetDevice(deviceIndex));
}

CudaDevice::CudaDevice(int deviceIndex) : deviceIndex(deviceIndex) 
{
	CUDA_CHECK(cudaGetDeviceProperties(&properties, deviceIndex));
	CUDA_CHECK(cudaSetDevice(deviceIndex));
}

CudaDevice::CudaDevice(const CudaDevice& cdv)
{
	properties = cdv.properties;
	deviceIndex = cdv.deviceIndex;
}

CudaDevice& CudaDevice::operator=(const CudaDevice& cdv)
{
	properties = cdv.properties;
	deviceIndex = cdv.deviceIndex;
	
	return *this;
}

ostream& operator<<(ostream& os, const CudaDevice& dv)
{
	int			   driverVersion,
				   runtimeVersion;
	cudaDeviceProp properties = dv.properties;

	CUDA_CHECK( cudaDriverGetVersion(&driverVersion) );
	CUDA_CHECK( cudaRuntimeGetVersion(&runtimeVersion) );

	os << "GPU-Name:                         " << properties.name << "\n";
	os << "CUDA Driver Version:              " << driverVersion << "\n";
	os << "CUDA Runtime Version:             " << runtimeVersion << "\n";
	os << "Grid Size Dim 1:                  " << properties.maxGridSize[0] << " bytes" << "\n";
	os << "Grid Size Dim 2:                  " << properties.maxGridSize[1] << " bytes" << "\n";
	os << "Grid Size Dim 3:                  " << properties.maxGridSize[2] << " bytes" << "\n";
	os << "Total Count of Threads per Block: " << properties.maxThreadsPerBlock << " Threads" << "\n";
	os << "Total Global Mem:                 " << properties.totalGlobalMem << " bytes" << "\n";
	os << "Total Const Mem:                  " << properties.totalConstMem << " bytes" << "\n";
	os << "shared Mem per block:             " << properties.sharedMemPerBlock << " bytes" << "\n";
	os << "Warp Size:                        " << properties.warpSize << " Threads" << "\n";
	return os;
}
