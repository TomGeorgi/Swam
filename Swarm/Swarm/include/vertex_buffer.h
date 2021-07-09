#pragma once

/*!
 * @brief VertexBuffer class is used to simply create & bind/unbind an OpenGL Vertex Buffer.
 */
class VertexBuffer
{
private:
	unsigned int renderID_;	//!< Holds the id of the created VertexBuffer.

public:

	/*!
	 * @brief Create an OpenGL VertexBuffer with the given data.
	 * @param data pointer to the data which should be hold in the OpenGL vertex buffer.
	 * @param size Size ot the data memory.
	 */
	VertexBuffer( const void* data, unsigned int size );

	/*!
	 * @brief Destroy Instance and delete OpenGL Buffer.
	 */
	~VertexBuffer();

	/*!
	 * @brief Bind Buffer.
	 */
	void bind() const;

	/*!
	 * @brief Unbind Buffer.
	 */
	void unbind() const;

	/*!
	 * @brief Get ID of the Buffer.
	 * @return number of the buffer / renderID.
	 */
	inline const unsigned int getBufferID() const { return renderID_; }
};