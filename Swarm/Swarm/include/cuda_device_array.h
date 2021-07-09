#pragma once

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

#include "macros.h"


/*!
 * @brief CudaDeviceArray Class can be used to create memory with any type easily on both CPU and GPU.
 *		  The allocated memory can be copied with the given methods to and from the GPU.
 * @tparam T data type
 */
template <class T> 
class CudaDeviceArray 
{
public:
	
	/*!
	 * @brief Standard Constructor. Set size to 0
	 */
	explicit CudaDeviceArray() 
		: start_(0), end_(0)
	{}

	/*!
	 * @brief Constructor to create a DeviceArray with a given size on the GPU.
	 * @param size Allocate the given value on GPU
	 */
	explicit CudaDeviceArray(size_t size)
	{
		allocate(size);
	}
	
	/*!
	 * @brief Destructor. Free the allocated memory on GPU.
	 */
	~CudaDeviceArray()
	{
		free();
	}

	/*!
	 * @brief Equals Operator is used to Copy an instanced CudaDeviceArray to a new instance.
	 * @param cdva CudaDeviceArray instance.
	 * @return new CudaDeviceArray instance.
	 */
	CudaDeviceArray& operator=( const CudaDeviceArray& cdva )
	{
		start_ = cdva.start_;
		end_ = cdva.end_;

		return *this;
	}

	/*!
	 * @brief Resize Array on GPU.
	 * @param size new size of Array.
	 */
	void resize(size_t size)
	{
		free();
		allocate(size);
	}

	/*!
	 * @brief Get Size of Array
	 * @return size of array.
	 */
	size_t getSize() const
	{
		return end_ - start_;
	}

	/*!
	 * @brief Get Data in array.
	 * @return data.
	 */
	const T* getData() const
	{
		return start_;
	}

	/*!
	 * @brief Get Data in array.
	 * @return data.
	 */
	T* getData() 
	{
		return start_;
	}

	/*!
	 * @brief Copy the given data to the GPU Memory.
	 * @param src source of data.
	 * @param size size of the given data.
	 */
	void set(const T* src, size_t size)
	{
		size_t min = std::min(size, getSize());
		cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
		if (result != cudaSuccess)
		{
			std::cerr << cudaGetErrorName(result) << ": " << cudaGetErrorString(result) << std::endl;
			throw std::runtime_error("failed to copy to device memory");
		}
	}

	/*!
	 * @brief Copy the data from the GPU Memory to the given destination pointer on the CPU.
	 * @param dest Destination to copy the data.
	 * @param size Size to copy.
	 */
	void get(T* dest, size_t size)
	{
		size_t min = std::min(size, getSize());
		//std::cout << min << std::endl;
		cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
		if (result != cudaSuccess)
		{
			std::cerr << cudaGetErrorName(result) << ": " << cudaGetErrorString(result) << std::endl;
			throw std::runtime_error("failed to copy to host memory!");
		}
	}

private:

	/*!
	 * @brief Allocate memory on the GPU.
	 * @param size How much memory shall be allocated.
	 */
	void allocate(size_t size)
	{
		cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(T));
		if (result != cudaSuccess)
		{
			start_ = end_ = 0;
			std::cerr << cudaGetErrorName(result) << ": " << cudaGetErrorString(result) << std::endl;
			throw std::runtime_error("failed to allocate device memory");
		}
		end_ = start_ + size;
	}

	/*!
	 * @brief Free the allocated memory.
	 */
	void free()
	{
		if (start_ != 0)
		{
			cudaFree(start_);
			start_ = end_ = 0;
		}
	}

	T* start_;	//!< start of the CudaDeviceMemory.
	T* end_;	//!< end of the CudaDeviceMemory.
};