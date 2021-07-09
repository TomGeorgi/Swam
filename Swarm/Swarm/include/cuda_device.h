#pragma once
#include <iostream>

#include <glew.h>
#include <glfw3.h>

#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"

#include "macros.h"
#include "vertex_buffer.h"

using namespace std;

/*!
 * @brief CudaDeivce class is used to simply manipulate a NVIDIA GPU.
 */
class CudaDevice
{
private:

	cudaDeviceProp properties;						//!< CudaDeviceProperties. Can be used for informations about the GPU.
	int deviceIndex;								//!< Index of device which should be handled by instance (Standard is 0).

	cudaGraphicsResource* cuda_vbo_resource = NULL;	//!< CudaGraphicsResouce. Is used to manipulate an OpenGL VertexBuffer directly via CUDA.

public:

	/*!
	 * @brief Standard Constructor. Initialize instance with deviceIndex 0, 
	 *		  to manipulate the first found CUDA device.
	 */
	CudaDevice();

	/*!
	 * @brief Initialize CudaDevice instance with a prefered deviceIndex.
	 * @param deviceIndex GPU index.
	 */
	CudaDevice(int deviceIndex);

	/*!
	 * @brief Initialize CudaDevice instance with a prefered index and predefined properties (unused).
	 * @param deviceIndex GPU index.
	 * @param properties GPU properties.
	 */
	CudaDevice(int deviceIndex, cudaDeviceProp properties);

	/*!
	 * @brief Copy Constructor
	 * @param cdv CudaDevice instance to copy.
	  */
	CudaDevice(const CudaDevice& cdv);

	/*!
	 * @brief CudaDevice destructor.
	 */
	~CudaDevice() {};

	/*!
	 * @brief Register an OpenGL Vertex Buffer. This Buffer will be manipulated directly via CUDA.
	 * @param vb OpenGL VertexBuffer.
	 */
	void registerGLBuffer( VertexBuffer vb );

	/*!
	 * @brief Unregister OpenGL Vertex Buffer source.
	 */
	void unregisterGLBuffer();

	/*!
	 * @brief Map OpenGL Buffer Resource to get access to this directly via CUDA.
	 */
	void mapResources();

	/*!
	 * @brief Unmap OpenGL Buffer Resource.
	 */
	void unmapResources();

	/*!
	 * @brief Returns a device pointer to the OpenGL Buffer (Must be mapped!).
	 * @param dev_ptr Device pointer will point to Buffer in CUDA.
	 * @param size size of buffer.
	 */
	void getMappedPointer( void** dev_ptr, size_t* size );

	/*!
	 * @brief Returns the number of processors on the GPU.
	 * @return number of processors.
	 */
	int getNumProcessors();

	/*!
	 * @brief Get the GPU device index.
	 * @return index of GPU in instance.
	 */
	inline int getDevice() { return deviceIndex; }

	/*!
	 * @brief Get Cuda Device Properties.
	 * @return Properties of Cuda Device.
	*/
	cudaDeviceProp getProperties();

	/*!
	 * @brief Set Cuda instance via equals.
	 * @param cdv CudaDevice instance
	 * @return new CudaDevice instance
	 */
	CudaDevice& operator=(const CudaDevice& cdv);

	/*!
	 * @brief Print some information about the Cuda Device with iostream.
	 * @param os stream
	 * @param dv Device to print
	 * @return stream with information about the device.
	 */
	friend ostream& operator<<(ostream& os, const CudaDevice& dv);
};



