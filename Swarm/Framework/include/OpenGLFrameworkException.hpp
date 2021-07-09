#ifndef OPENGLFRAMEWORKEXCEPTION_HPP
#define OPENGLFRAMEWORKEXCEPTION_HPP

#include <exception>
#include <string>

class OpenGLFrameworkException : public std::exception
{
public:
	OpenGLFrameworkException( std::string const & _message );
};

#endif