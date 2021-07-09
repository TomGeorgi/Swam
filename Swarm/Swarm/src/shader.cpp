#include <glew.h>
#include <fstream>
#include <sstream>
#include <iostream>

#include "shader.h"

Shader::Shader( const std::string& _vertexFilePath, const std::string& _fragmentFilePath )
{
	ShaderProgramSource source = ParseShader( _vertexFilePath, _fragmentFilePath );
	renderID_ = createShader( source.VertexSource, source.FragmentSource );
}

Shader::~Shader()
{
	glDeleteProgram( renderID_ );
}

void Shader::bind() const
{
	glUseProgram( renderID_ );
}

void Shader::unbind() const
{
	glUseProgram( 0 );
}

void Shader::setUniform1i( const std::string& _name, int _value )
{
	glUniform1i( getUniformLocation( _name ), _value );
}

void Shader::setUniform1f( const std::string& _name, float _value )
{
	glUniform1f( getUniformLocation( _name ), _value );
}

void Shader::setUniformMat4f( const std::string& _name, const glm::mat4& _matrix )
{
	glUniformMatrix4fv( getUniformLocation( _name ), 1, GL_FALSE, &_matrix[0][0] );
}



Shader::ShaderProgramSource Shader::ParseShader( const std::string& _vertexFilePath, const std::string& _fragmentFilePath )
{
	// Read the vertex shader code from file
	std::string vertexShaderCode;
	std::ifstream vertexShaderStream( _vertexFilePath, std::ios::in );
	
	if ( vertexShaderStream.is_open() )
	{
		std::stringstream sstr;
		sstr << vertexShaderStream.rdbuf();
		vertexShaderCode = sstr.str();
		vertexShaderStream.close();
	} else
	{
		std::cout << "Impossible to open " << _vertexFilePath << "!" << std::endl; ;
		return {};
	}

	// Read the fragment shader code from file
	std::string fragmentShaderCode;
	std::ifstream fragmentShaderStream( _fragmentFilePath, std::ios::in );
	if ( fragmentShaderStream.is_open() )
	{
		std::stringstream sstr;
		sstr << fragmentShaderStream.rdbuf();
		fragmentShaderCode = sstr.str();
		fragmentShaderStream.close();
	} else
	{
		std::cout << "Impossible to open " << _fragmentFilePath << "!" << std::endl; ;
		return {};
	}

	return { vertexShaderCode, fragmentShaderCode };
}

unsigned int Shader::compileShader( unsigned int _type, const std::string& _source )
{
	unsigned int id = glCreateShader( _type );
	const char* src = _source.c_str();
	glShaderSource( id, 1, &src, nullptr );
	glCompileShader( id );

	int result;
	glGetShaderiv( id, GL_COMPILE_STATUS, &result );
	if ( result == GL_FALSE )
	{
		int length;
		glGetShaderiv( id, GL_INFO_LOG_LENGTH, &length );
		
		std::vector<char> message( length + 1 );
		glGetShaderInfoLog( id, length, &length, message.data() );

		std::cout << "Failed to compile " << ( _type == GL_VERTEX_SHADER ? "vertex" : "fragment" ) << " shader!" << std::endl;
		std::cout << message.data() << std::endl;
		return 0;
	}

	return id;
}

unsigned int Shader::createShader( const std::string& _vertexShader, const std::string& _fragmentShader )
{
	unsigned int program = glCreateProgram();
	unsigned int vs = compileShader( GL_VERTEX_SHADER, _vertexShader );
	unsigned int fs = compileShader( GL_FRAGMENT_SHADER, _fragmentShader );

	glAttachShader( program, vs );
	glAttachShader( program, fs );
	glLinkProgram( program );
	glValidateProgram( program );

	glDeleteShader( vs );
	glDeleteShader( fs );

	return program;
}

int Shader::getUniformLocation( const std::string& name )
{
	if ( uniformLocationCache_.find( name ) != uniformLocationCache_.end() )
		return uniformLocationCache_[name];

	int location = glGetUniformLocation( renderID_, name.c_str() );
	if ( location == -1 )
		std::cout << "Warning: uniform '" << name << "' doesn't exist" << std::endl;

	uniformLocationCache_[name] = location;
	return location;
}
