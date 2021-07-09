#pragma once

#include "vertex_buffer_layout.h"
#include "vertex_buffer.h"

/*!
 * @brief VertexArray Class is used to simply create, manipulate & bind/unbind an OpenGL Vertex Array.
 */
class VertexArray
{
private:
	unsigned int renderID_;	//!< Holds the id of the created VertexArray.

public:

	/*!
	 * @brief Standard Constructor. Generates a new OpenGL VertexArray.
	 */
	VertexArray();

	/*!
	 * @brief Destroy OpenGL VertexArray.
	 */
	~VertexArray();

	/*!
	 * @brief Add a Buffer to Vertex Array.
	 * @param _vb VertexBuffer.
	 * @param _layout VertexBufferLayout. Contains the information how to handle the data in the vertex buffer.
	 */
	void addBuffer( const VertexBuffer& _vb, const VertexBufferLayout& _layout );

	/*!
	 * @brief Add a Buffer to Vertex Array with a given index.
	 * @param _vb VertexBuffer.
	 * @param _vbElement VertexBufferElement contains informations about how to handle the elements in the vertex buffer.
	 * @param _vertexIndex Specifies the index of the generic vertex attribute to be modified.
	 */
	void addBuffer( const VertexBuffer& _vb, const VertexBufferElement _vbElement, int _vertexIndex );

	/*!
	 * @brief Bind Vertex Array.
	 */
	void bind() const;

	/*!
	 * @brief Unbind Vertex Array.
	 */
	void unbind() const;
};