#include "OpenGLFrameworkException.hpp"

/**
	Constructor.

	@param _message The message for the exception.
*/
OpenGLFrameworkException::OpenGLFrameworkException( std::string const & _message ) :
	std::exception( _message.c_str() )

{

}
