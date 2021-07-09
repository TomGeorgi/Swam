#pragma once

#include "glew.h"
#include <vector>
#include <cstdlib>

/*!
 * @brief Struct of an VertexBufferElement.
 */
struct VertexBufferElement
{
	unsigned int type;			//!< DataType in Buffer
	unsigned int count;			//!< Number of elements for One Vertex in Buffer. (For Example 3 = triangle, ...)
	unsigned int normalized;	//!< Is the data normalized
	unsigned int offset;		//!< Offset. Where should begins this type of vertexbufferelemty
};

/*!
 * @brief VertexBufferLayout class is used to give a vertexbuffer a structure (mulitple buffers in one array).
 */
class VertexBufferLayout
{
private:
	std::vector<VertexBufferElement> elements_;	//!< structure elements.

public:

	/*!
	 * @brief Standard Constructor.
	 */
	VertexBufferLayout() = default;

	/*!
	 * @brief push a new VertexBufferElement Structure.
	 * @tparam T data type
	 * @param _count Number of elements for One Vertex in Buffer.
	 * @param _offset Offset. Where should begins this type of vertexbufferelemty
	 */
	template<typename T>
	void push( unsigned int _count, unsigned int _offset )
	{
		static_assert( true, "Error should not reach this statement" );
	}

	/*!
	 * @brief get VertexBufferLayout elements
	 * @return vector of VertexBufferElements.
	 */
	inline const std::vector<VertexBufferElement>& getElements() const { return elements_; }
};

template<> inline
void VertexBufferLayout::push<float>( unsigned int _count, unsigned int _offset )
{
	elements_.push_back( { GL_FLOAT, _count, GL_FALSE, _offset } );
}

template<> inline
void VertexBufferLayout::push<unsigned int>( unsigned int _count, unsigned int _offset )
{
	elements_.push_back( { GL_UNSIGNED_INT, _count, GL_FALSE, _offset } );
}

template<> inline
void VertexBufferLayout::push<unsigned char>(unsigned int _count, unsigned int _offset)
{
	elements_.push_back( { GL_UNSIGNED_BYTE, _count, GL_TRUE, _offset } );
}
