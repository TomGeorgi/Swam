#include "Window.hpp"
#include <iostream>
#include <glew.h>
#include <glfw3.h>

/**
	The window instance must be initialized with null.
*/
Window * Window::m_instance = NULL;

/**
	Initializes the window instance and returns always the same object.

	@return Returns the window instance.
*/
Window * Window::getInstance()
{
	if( NULL == m_instance )
	{
		m_instance = new Window();
	}

	return m_instance;
}

/**
	Constructor.
	Initializes the window and the context ressources.
*/
Window::Window() :
	m_window( NULL ),
	m_camera( CameraType::PERSPECTIVE )
{
	currentTime_ = getCurrentTime();
	lastUpdate_ = getCurrentTime();

	glewExperimental = GL_TRUE;
	if( GL_FALSE == glfwInit() )
	{
		std::cerr << "Unable to initialize glfw library." << std::endl;
	}

	else
	{
		m_camera.setDistancePlanes( 1.0f, 10000.0f );
	}
}

/**
	Destructor.
	Closes the window and frees all allocated ressources.
*/
Window::~Window()
{
	close();

	glfwTerminate();
}

void Window::setWindowTitle( const char* _title )
{
	if ( isOpen() )
	{ 
		glfwSetWindowTitle( m_window, _title );
	}
}

/**
	Opens the window with the passed pixel width and height.

	@param _title The title for the window.
	@param _pixelWidth The width of the window in pixels.
	@param _pixelHeight The height of the window in pixels.
*/
void Window::open( std::string const & _title,
				   GLuint const & _pixelWidth,
				   GLuint const & _pixelHeight)
{
	if( !isOpen() )
	{
		// Set GLFW error callback
		glfwSetErrorCallback( errorCallback );

		glfwWindowHint( GLFW_SAMPLES, 4 );
		glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
		glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
		glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

		m_window = glfwCreateWindow( static_cast< int >( _pixelWidth ),
									 static_cast< int >( _pixelHeight ),
									 _title.c_str(), NULL, NULL );
		setActive();
		
		GLenum const glewError = glewInit();
		if( GLEW_OK != glewError )
		{
			close();
			std::cerr << "Unable to initialize glew library: ";
			std::cerr << glewGetErrorString( glewError ) << std::endl;
		}

		else
		{
			std::cout << "Glew is initialized for window " << _title.c_str() << "." << std::endl;
		}

		if( !glewIsSupported( "GL_VERSION_3_3" ) )
		{
			close();
			std::cerr << "OpenGL 3.3 is not supported!" << std::endl;
		}

		else
		{
			std::cout << "OpenGL 3.3 is supported." << std::endl;
		}

		m_camera.setWindowSize( static_cast< GLfloat > ( _pixelWidth ),
								static_cast< GLfloat >( _pixelHeight ) );
		glfwSetKeyCallback( m_window, Window::handleKeyEvent );
		glfwSetWindowSizeCallback( m_window, Window::handleResizeEvent );
		glfwSetFramebufferSizeCallback( m_window, Window::handleFramebufferResizeEvent );

		// Set Scroll Callback
		glfwSetScrollCallback( m_window, Window::scrollCallback );

		/* Enable / Disable V-Sync */
		glfwSwapInterval( 1 );

		// Enable opengl debug callback
		glEnable( GL_DEBUG_OUTPUT );
		glEnable( GL_DEBUG_OUTPUT_SYNCHRONOUS );
		glDebugMessageCallback( openglErrorCallback, NULL );
		glDebugMessageControl( GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, static_cast< GLboolean >( true ) );

		windowTitle_ = _title;
	}
}

/**
	@return	Returns true, if the window is open. False, otherwise.
*/
GLboolean Window::isOpen() const
{
	return NULL != m_window;
}

/**
	Closes the window.
	You can reopen the window by calling open().
*/
void Window::close()
{
	if( isOpen() )
	{
		glfwSetKeyCallback( m_window, NULL );
		glfwSetWindowSizeCallback( m_window, NULL );
		glfwSetFramebufferSizeCallback( m_window, NULL );

		glfwSetWindowShouldClose( m_window, GL_FALSE );
		
		m_window = NULL;
	}
}

/**
	Sets the new eye point of the window camera to the passed point.

	@param _eyePoint The new eye point.
*/
void Window::setEyePoint( glm::vec4 const & _eyePoint )
{
	m_camera.setEyePoint( _eyePoint );
}

/**
	@return Returns the width of the window.
*/
GLuint Window::getWidth() const
{
	if( isOpen() )
	{
		int width = 0;
		glfwGetWindowSize( m_window, &width, NULL );
		return static_cast< GLuint >( width );
	}

	else
	{
		return 0;
	}
}

/**
	@return Returns the window height.
*/
GLuint Window::getHeight() const
{
	if( isOpen() )
	{
		int height = 0;
		glfwGetWindowSize( m_window, NULL, &height );
		return static_cast< GLuint >( height );
	}

	else
	{
		return 0;
	}
}

/**
	Activates the current window context to be used by OpenGL.
*/
void Window::setActive()
{
	if( isOpen() )
	{
		glfwMakeContextCurrent( m_window );
	}
}

/**
	Draws the current buffer to the monitor and clears the old buffer to be
	ready for rendering.
*/
void Window::swapBuffer()
{
	if( isOpen() )
	{
		setActive();

		glfwSwapBuffers( m_window );

		glfwPollEvents();

		if( GL_TRUE == glfwWindowShouldClose( m_window ) )
		{
			close();
		}
	}
}

/**
	@return Returns a copy of the camera instance.
*/
Camera Window::getCamera() const
{
	return Camera( m_camera );
}

/**
	GLFW keyboard callback function.

	@param _window The window instance of the key event.
	@param _key The key enumeration ( GLFW_KEY_* ).
	@param _scancode Unused.
	@param _action The action of the event ( GLFW_PRESS or GLFW_RELEASE ).
	@param _mods Mods ( GLFW_MOD_SHIFT, GLFW_MOD_CONTROL, GLFW_MOD_ALT, GLFW_MOD_SUPER ).
*/
void Window::handleKeyEvent( GLFWwindow * _window,
							 GLint _key,
							 GLint _scancode,
							 GLint _action,
							 GLint _mods )
{
	Window * const window = getInstance();
	window->handleKeyEvent( _key, _action, _mods );
}

/**
	GLFW window resize callback function.

	@param _window The window instance of the resize event.
	@param _width The new width of the window.
	@param _height The new height of the window.
*/
void Window::handleResizeEvent( GLFWwindow * _window,
								int _width,
								int _height )
{
	Window * const window = getInstance();
	window->handleResizeEvent( static_cast< GLuint >( _width ),
							   static_cast< GLuint >( _height ) );
}

/**
	Callback function to resize the framebuffer of the passed window instance.

	@param	_window	The window for the resize event.
	@param	_width	The new framebuffer width in pixels.
	@param	_height	The new framebuffer height in pixels.
*/
void Window::handleFramebufferResizeEvent( GLFWwindow * _window, int _width, int _height )
{
	Window * const window = getInstance();
	window->handleFramebufferResizeEvent( static_cast< GLsizei >( _width ),
										  static_cast< GLsizei >( _height ) );
}

/**
	Callback function to handle key events for this window.

	@param	_key	The key to handle.
	@param	_action	The action. For example GLFW_PRESS or GLFW_REPEAT.
*/
void Window::handleKeyEvent( GLint const & _key,
							 GLint const & _action,
							 GLint const & )
{
	if( isOpen() )
	{
		if( GLFW_PRESS == _action || GLFW_REPEAT == _action )
		{
			// std::cout << "Handle key " << _key << "." << std::endl;

			switch( _key )
			{
				case GLFW_KEY_W:
					m_camera.translateEyePoint( m_camera.viewDirection() * 10.0f );
					break;

				case GLFW_KEY_S:
					m_camera.translateEyePoint( m_camera.viewDirection() * -10.0f );
					break;

				case GLFW_KEY_A:
					m_camera.translateEyePoint( m_camera.horizontalDirection() * 10.0f );
					break;

				case GLFW_KEY_D:
					m_camera.translateEyePoint( m_camera.horizontalDirection() * -10.0f );
					break;

				case GLFW_KEY_Q:
					m_camera.translateEyePoint( m_camera.upDirection() * 10.0f );
					break;

				case GLFW_KEY_E:
					m_camera.translateEyePoint( m_camera.upDirection() * -10.0f );
					break;

				case GLFW_KEY_LEFT:
					m_camera.rotateYaw( 5.0f );
					break;

				case GLFW_KEY_RIGHT:
					m_camera.rotateYaw( -5.0f );
					break;

				case GLFW_KEY_UP:
					m_camera.rotatePitch( 5.0f );
					break;

				case GLFW_KEY_DOWN:
					m_camera.rotatePitch( -5.0f );
					break;

				case GLFW_KEY_R:
					m_camera.setEyePoint( glm::vec4( 0.0f, 0.0f, 500.0f, 1.0f ) );
					m_camera.resetAngles();
					break;

				default:
					break;
			}
		}
	}
}

/**
	Callback function for the resize event of this window.

	@param	_width	The new width of the window.
	@param	_height	The new height of the window.
*/
void Window::handleResizeEvent( GLuint const & _width,
								GLuint const & _height )
{
	if( isOpen() )
	{
		std::cout << "Window resized." << std::endl;
		m_camera.setWindowSize( static_cast< GLfloat >( _width ),
								static_cast< GLfloat >( _height ) );
	}
}

/**
	Callback function for the framebuffer resize event without window instance.

	@param	_width	The new framebuffer size in pixels.
	@param	_height	The new framebuffer size in pixels.
*/
void Window::handleFramebufferResizeEvent( GLsizei const & _width, GLsizei const & _height )
{
	if( isOpen() )
	{
		std::cout << "Framebuffer resized." << std::endl;
		glViewport( 0, 0, _width, _height );
	}
}

void Window::open( std::string const& _title )
{
	open( _title, WIDTH, HEIGHT );
}

void Window::updateDisplay()
{
	swapBuffer();
	computeFPS();
}

double Window::getCurrentTime()
{
	return glfwGetTime();
}

Window::CursorPosition Window::getCursorPos()
{
	double x, y;
	glfwGetCursorPos( m_window, &x, &y );
	return { x, y };
}

void Window::openglErrorCallback( GLenum _source, GLenum _type, GLenum id, GLenum severity, GLsizei _length, const GLchar* _message, const void* _userParam )
{
	// ignore non-significant error/warning codes
	if ( id == 131169 || id == 131185 || id == 131218 || id == 131204 ) return;

	std::cout << "---------------" << std::endl;
	std::cout << "Debug message (" << id << "): " << _message << std::endl;

	switch ( _source )
	{
		case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
		case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
		case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
		case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
		case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
		case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
	} std::cout << std::endl;

	switch ( _type )
	{
		case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
		case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
		case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
		case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
		case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
		case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
		case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
	} std::cout << std::endl;

	switch ( severity )
	{
		case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high";
											 std::cerr << "Aborting..." << std::endl;
											 abort();
		case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
		case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
		case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
	} std::cout << std::endl;
	std::cout << std::endl;
}

void Window::errorCallback( int _error, const char* _description )
{
	std::cerr << "Error: " << _description << std::endl;
}

void Window::scrollCallback( GLFWwindow * window, double xoffset, double yoffset )
{
	
}

void Window::computeFPS()
{
	fpsCount_++;
	currentTime_ = getCurrentTime();

	double timeDiff = currentTime_ - lastUpdate_;
	if ( timeDiff >= 1.0 )
	{
		char fps[256];
		float framesPerms = ( timeDiff * 1000 ) / double( fpsCount_ );
		float ifps = double( fpsCount_ ) / timeDiff;

		sprintf_s( fps, "%s: %3.1f FPS || %3.3f ms/frame", windowTitle_.c_str(), ifps, framesPerms );
		setWindowTitle( fps );

		fpsCount_ = 0;
		lastUpdate_ = getCurrentTime();
	}
}
