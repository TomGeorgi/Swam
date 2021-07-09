#include "vertex_array.h"

VertexArray::VertexArray()
{
	glGenVertexArrays( 1, &renderID_ );
}

VertexArray::~VertexArray()
{
	glDeleteVertexArrays( 1, &renderID_ );
}

void VertexArray::addBuffer( const VertexBuffer& _vb, const VertexBufferLayout& _layout )
{
	bind();
	_vb.bind();

	const auto& elements = _layout.getElements();
	for ( unsigned int i = 0; i < elements.size(); ++i )
	{
		const auto& element = elements[i];
		glEnableVertexAttribArray( i );
		glVertexAttribPointer( i, element.count, element.type, element.normalized, 0, reinterpret_cast< const void* >( element.offset ) );
	}
}

void VertexArray::addBuffer( const VertexBuffer& _vb, const VertexBufferElement _vbElement, int _vertexIndex )
{
	bind();
	_vb.bind();

	glEnableVertexAttribArray( _vertexIndex );
	glVertexAttribPointer( _vertexIndex, _vbElement.count, _vbElement.type, _vbElement.normalized, 0, reinterpret_cast< const void* >( _vbElement.offset ) );
}

void VertexArray::bind() const
{
	glBindVertexArray( renderID_ );
}

void VertexArray::unbind() const
{
	glBindVertexArray( 0 );
}




