#include <glew.h>
#include "vertex_buffer.h"

VertexBuffer::VertexBuffer( const void* data, unsigned int size )
{
	glGenBuffers( 1, &renderID_ );
	glBindBuffer( GL_ARRAY_BUFFER, renderID_ );
	glBufferData( GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW );
}

VertexBuffer::~VertexBuffer()
{
	glDeleteBuffers( 1, &renderID_ );
}

void VertexBuffer::bind() const
{
	glBindBuffer( GL_ARRAY_BUFFER, renderID_ );
}

void VertexBuffer::unbind() const
{
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
}
