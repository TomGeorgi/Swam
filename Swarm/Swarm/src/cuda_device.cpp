#include "cuda_device.h"

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

CudaDevice::CudaDevice( int deviceIndex ) : deviceIndex( deviceIndex )
{
	CUDA_CHECK( cudaGetDeviceProperties( &properties, deviceIndex ) );
	CUDA_CHECK( cudaSetDevice( deviceIndex ) );
}

CudaDevice::CudaDevice(int deviceIndex, cudaDeviceProp propertie)
	: deviceIndex(deviceIndex), properties( properties )
{
	CUDA_CHECK(cudaGetDeviceProperties(&properties, deviceIndex));
	CUDA_CHECK(cudaSetDevice(deviceIndex));
}

CudaDevice::CudaDevice(const CudaDevice& cdv)
{
	properties = cdv.properties;
	deviceIndex = cdv.deviceIndex;
}

void CudaDevice::registerGLBuffer( VertexBuffer vb )
{
	CUDA_CHECK( cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vb.getBufferID(), cudaGraphicsMapFlagsNone ) );
}

void CudaDevice::unregisterGLBuffer()
{
	CUDA_CHECK( cudaGraphicsUnregisterResource( cuda_vbo_resource ) );
	cuda_vbo_resource = NULL;
}

void CudaDevice::mapResources()
{
	CUDA_CHECK( cudaGraphicsMapResources( 1, &cuda_vbo_resource, NULL ) );
}

void CudaDevice::unmapResources()
{
	CUDA_CHECK( cudaGraphicsUnmapResources( 1, &cuda_vbo_resource, 0 ) );
}

void CudaDevice::getMappedPointer(void **dev_ptr, size_t* size)
{
	CUDA_CHECK( cudaGraphicsResourceGetMappedPointer( dev_ptr, size, cuda_vbo_resource ) );
}

int CudaDevice::getNumProcessors()
{
	int numProc;
	CUDA_CHECK(cudaDeviceGetAttribute( &numProc, cudaDevAttrMultiProcessorCount, deviceIndex ));
	return numProc;
}

cudaDeviceProp CudaDevice::getProperties()
{
	return properties;
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
