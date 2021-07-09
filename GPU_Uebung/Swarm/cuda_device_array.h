#pragma once

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

#include "macros.h"

template <class T> 
class DeviceArray 
{
public:

	explicit DeviceArray() 
		: start_(0), end_(0)
	{}

	explicit DeviceArray(size_t size)
	{
		allocate(size);
	}

	~DeviceArray()
	{
		free();
	}

	void resize(size_t size)
	{
		free();
		allocate(size);
	}

	size_t getSize() const
	{
		return end_ - start_;
	}

	const T* getData() const
	{
		return start_;
	}

	T* getData() 
	{
		return start_;
	}

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

	void get(T* dest, size_t size)
	{
		size_t min = std::min(size, getSize());
		std::cout << min << std::endl;
		cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
		if (result != cudaSuccess)
		{
			std::cerr << cudaGetErrorName(result) << ": " << cudaGetErrorString(result) << std::endl;
			throw std::runtime_error("failed to copy to host memory!");
		}
	}

private:

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

	void free()
	{
		if (start_ != 0)
		{
			cudaFree(start_);
			start_ = end_ = 0;
		}
	}

	T* start_;
	T* end_;
};