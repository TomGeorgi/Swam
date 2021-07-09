#pragma once

#include <string>
#include <unordered_map>
#include <glm/glm.hpp>

/*!
 * @brief Represent a Shader.
 */
class Shader
{
private:

	unsigned int renderID_;										//!< ID of shader program
	std::unordered_map<std::string, int> uniformLocationCache_; //!< Cache for uniform locations in shader

public:

	/*!
	 * @brief Constructor.
	 * @param _vertexFilePath path to vertex file
	 * @param _fragmentFilePath path to fragment file
	 */
	Shader( const std::string& _vertexFilePath, const std::string& _fragmentFilePath );

	/*!
	 * @brief Destructor. Destroys shader program.
	 */
	~Shader();

	/*!
	 * @brief Bind shader program.
	 */
	void bind() const;

	/*!
	 * @brief Unbind shader program.
	 */
	void unbind() const;

	/*!
	 * @brief Set integer value to uniform
	 * @param _name uniform name
	 * @param _value value
	 */
	void setUniform1i( const std::string& _name, int _value );

	/*!
	 * @brief Set float value to uniform
	 * @param _name uniform name
	 * @param _value value
	 */
	void setUniform1f( const std::string& _name, float _value );

	/*!
	 * @brief Set matrix to uniform
	 * @param _name uniform name
	 * @param _matrix matrix
	 */
	void setUniformMat4f( const std::string& _name, const glm::mat4& _matrix );

private:

	/*!
	 * @brief Struct for Shader Program source.
	 */
	struct ShaderProgramSource
	{
		std::string VertexSource;
		std::string FragmentSource;
	};

	/*!
	 * @brief Parse shaders
	 * @param _vertexFilePath path to vertex file.
	 * @param _fragmentFilePath path to fragment file.
	 * @return ShaderProgramSource with vertex source and fragment source.
	 */
	ShaderProgramSource ParseShader( const std::string& _vertexFilePath, const std::string& _fragmentFilePath );

	/*!
	 * @brief compile shader
	 * @param _type shader type. Either vertex or fragment.
	 * @param _source program source.
	 * @return shader id.
	 */
	unsigned int compileShader( unsigned int _type, const std::string& _source );

	/*!
	 * @brief create shader 
	 * @param _vertexShader vertex shader source
	 * @param _fragmentShader fragment shader source
	 * @return render / program id
	 */
	unsigned int createShader( const std::string& _vertexShader, const std::string& _fragmentShader );

	/*!
	 * @brief get location of Uniform in shader
	 * @param name of uniform.
	 * @return uniform location.
	 */
	int getUniformLocation( const std::string& name );
};