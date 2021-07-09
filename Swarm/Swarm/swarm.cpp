#include "Window.hpp"

#include <vector>
#include <fstream>
#include <sstream>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>


int main( int _argc, char** _argv )
{
	// Create / Get Window instance
	Window* window = Window::getInstance();

	window->open( "Fish Swarm", 1200, 800 );
	window->setEyePoint( glm::vec4( 0.0f, 0.0f, 500.0f, 1.0f ) );
	window->setActive();

	initializeOpenGL()

	clock_t lastInterval = clock();

	while ( window->isOpen() )
	{
		window->setActive();

		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

		// TODO
		//drawOpenGL( window, lastInterval );

		lastInterval = clock();

		window->swapBuffer();
	}
}