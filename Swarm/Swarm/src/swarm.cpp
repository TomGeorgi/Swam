#include "Window.hpp"
#include "renderer.h"
#include "vec3.h"
#include "cuda_device.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>


std::string windowTitle = "Fish Swarm";

/*!
 * @brief Main
 * @return 0
 */
int main()
{
	Window* window = Window::getInstance();

	window->open( windowTitle, 1600, 1200 );
	window->setEyePoint( glm::vec4( 0.0f, 0.0f, 1000.0f, 1.0f ) );
	window->setActive();

	Renderer renderer;

	while ( window->isOpen() )
	{
		renderer.render();

		window->updateDisplay();
		window->setActive();
	}

	renderer.cleanUp();

	return 0;
}
