#include "glew.h"

#include <iostream>
#include <WindowsNumerics.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Window.hpp"
#include "renderer.h"
#include "kernel.h"

#include <device_launch_parameters.h>

/*
 * BoxSize for Random Spawn.
 */
static float const MAX_X = 5;
static float const MIN_X = -MAX_X;
static float const MAX_Y = 5;
static float const MIN_Y = -MAX_Y;
static float const MAX_Z = 5;
static float const MIN_Z = -MAX_Z;

// Constant orange. Is used for random color generation for each particle.
static GLfloat const color[3] = { 200.0f / 255.0f, 117.0f / 255.0f, 26.0f / 255.0f };

/*!
 * @brief Random Function. Creates a random color value.
 * @return random float value
*/
float randC()
{
	float inverse_scale = 2;
	return float( rand() ) / float( RAND_MAX ) / inverse_scale + (1 - 1 / inverse_scale / 2);
}

/*!
 * @brief Creates a random float value between a and b.
 * @param a first barrier.
 * @param b second barrier.
 * @return random float value.
*/
float randf( float a, float b )
{
	float random = ( ( float ) rand() ) / ( float ) RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

Renderer::Renderer() : shader_( "vertex.glsl", "fragment.glsl" )				// Create Shader Program
{
	std::vector<Vector3> waypointVector											// Create vector with waypoints for fishies
	{
		// zig zag
		Vector3(-4.0, -3.0, 0.0),
		Vector3(4.0, -2.0, 0.0),
		Vector3(-4.0, -1.0, 0.0),
		Vector3(4.0, 2.0, 0.0),
		
		// Square
		//Vector3(-5.0, 5.0, 0.0),
		//Vector3(-5.0, -5.0, 0.0),
		//Vector3(5.0, -5.0, 0.0),
		//Vector3(5.0, 5.0, 0.0),

		// not moving
		//Vector3(0,0,0)
	};

	waypointList = new WaypointList( waypointVector );							// Initialize Linked Waypoint list.
	swarmCenter = waypointList->get();											// Get first swarm center for particles.

	glEnable( GL_BLEND );														// clean looking points.
	glEnable( GL_PROGRAM_POINT_SIZE );											// enable to set the point size.

	device_ = CudaDevice();														// Create CUDA Device. Automically select the first found device.
	std::cout << device_ << std::endl;											// Print out some information about the used GPU

	Window* window = Window::getInstance();										// Used to set current time

	
	createBuffers();															// create buffers related to OpenGL and CUDA
	setLastUpdate(window->getCurrentTime());
}

void Renderer::createBuffers()
{
	
	for ( unsigned int i = 0; i < NUM_PARTICLES; i++ )							// init vertex position and color
	{
		h_data.push_back( randf( MIN_X, MAX_X) );								// random vertex.x
		h_data.push_back( randf( MIN_Y, MAX_Y ) );								// random vertex.y
		h_data.push_back( randf( MIN_Z, MAX_Z ) );								// random vertex.z
		h_data.push_back( 1.0f );												// vertex.w

		h_color.push_back( color[0] * randC() );								// Red
		h_color.push_back( color[1] * randC() );								// Green
		h_color.push_back( color[2] * randC() );								// Blue
		h_color.push_back( 1.0f );												// Alpha
	}

	VertexBuffer vb( h_data.data(), NUM_PARTICLES * 4 * sizeof( float ) );		// Create buffer for positions
	VertexBuffer vbC( h_color.data(), NUM_PARTICLES * 4 * sizeof( float ) );	// Create buffer for colors

	VertexBufferLayout layout;													// Create Buffer Layout. Is used to call the VAO how to handle the buffers.
	layout.push<float>( 4, 0 );													// float values, 4 values per vertice and start at 0 (no offset).

	va_.addBuffer( vb, layout );												// Add 1. Buffer (Position). This buffer will be modified in kernel later.
	va_.addBuffer( vbC, layout.getElements()[0], 1 );							// Add 2. Buffer (Color). It's a little bit more complicated than the last line, because we need to add an index seperately.

	va_.unbind();																// Unbind VAO while unused.
	vb.unbind();																// Unbind VBO. Unused now.
	vbC.unbind();																// Unbind VBO. Unused now.

	device_.registerGLBuffer( vb );												// CUDA: register opengl buffer object for access CUDA

	for ( unsigned int i = 0; i < NUM_PARTICLES; i++ )							// init forces.
	{
		h_state.push_back(0.0f);												// force.x
		h_state.push_back(0.0f);												// force.y
		h_state.push_back(0.0f);												// force.z
		h_state.push_back(static_cast< float >( rand() ) / RAND_MAX / 4 + 0.875);	// random mass for each particle
	}

	h_shark_state = { 0.01, 0.01, 0.01, 0.01 };									// Shark forces and mass 

	/*
	 * Explicit creation and copy, because we won't update this values.
	 */
	d_state = new CudaDeviceArray<float>( NUM_PARTICLES * 4 );					// Allocate Memory on GPU for force vector
	d_state->set( h_state.data(), NUM_PARTICLES * 4 );							// Copy force vector to GPU
	d_color = new CudaDeviceArray<float>(NUM_PARTICLES * 4);					// Allocate Memory on GPU for color vector
	d_color->set(h_color.data(), NUM_PARTICLES * 4);							// Copy color vector to GPU 

	kernel_init_grid(NUM_PARTICLES);											// Initialize grid depending on the number of particles.


	// shark buffer
	h_shark_data.push_back(5.0f);
	h_shark_data.push_back(5.0f);
	h_shark_data.push_back(5.0f);
	h_shark_data.push_back(1.0f);

	// shark color
	h_shark_color.push_back(0.8f);
	h_shark_color.push_back(0.8f);
	h_shark_color.push_back(0.8f);
	h_shark_color.push_back(1.0f);

	VertexBuffer vbShark(h_shark_data.data(), NUM_SHARKS * 4 * sizeof(float));	// Shark Position VBO
	VertexBuffer vbSharkC(h_shark_color.data(), NUM_SHARKS * 4 * sizeof(float));// Shark Color VBO

	vaShark.addBuffer(vbShark, layout);											// Add Position VBO to VAO
	vaShark.addBuffer(vbSharkC, layout.getElements()[0], 1);					// Add Color VBO to VAO

	vaShark.unbind();															// Unbind VAO while unused.
	vbShark.unbind();															// Unbind VBO. Unused now.
	vbSharkC.unbind();															// Unbind VBO. Unused now.
}

void Renderer::runCuda()
{
	float4* vboPtr;
	size_t numBytes;

	device_.mapResources();														// Map Position VBO with CUDA.
	device_.getMappedPointer( ( void** ) &vboPtr, &numBytes );					// Get Pointer to memory.

	void* data = d_state->getData();											// Get Pointer to states

	// Call Kernel
	kernel_advance(
		vboPtr,
		static_cast<float4*>(data),
		NUM_PARTICLES,
		speed,
		swarmCenter,
		Vector3(h_shark_data[0], h_shark_data[1], h_shark_data[2]));

	device_.unmapResources();													// Unmap Resources while unused.
}

void Renderer::moveSwarmCenter()
{
	Vector3 diff = waypointList->get() - swarmCenter;							// Get Next Swarm center
	if (diff.length() < WAYPOINT_THRESHOLD)										// Check if center was reached
	{
		diff = waypointList->getNext() - swarmCenter;
	}

	diff = diff.normalized() * speed;
	swarmCenter += diff;
}

void Renderer::moveShark()
{
	Vector3 shark_data = { h_shark_data[0], h_shark_data[1], h_shark_data[2] };
	Vector3 shark_state = { h_shark_state[0], h_shark_state[1], h_shark_state[2] };
	Vector3 diff = swarmCenter - shark_data;

	// turn back to swarm
	if (diff.length() > 4.0)
	{
		diff = diff.normalized() * speed * 0.2;
		shark_state.x += diff.x;
		shark_state.y += diff.y;
		shark_state.z += diff.z;
		if (shark_state.length() > speed * 1.3)
		{
			shark_state = shark_state.normalized() * speed * 1.3;
		}
	}
	// swim through swarm or leave it
	else
	{
		if (shark_state.length() < speed * 3)
		{
			shark_state.x *= 1.1;
			shark_state.y *= 1.1;
			shark_state.z *= 1.1;
		}
	}

	h_shark_data[0] += shark_state.x;
	h_shark_data[1] += shark_state.y;
	h_shark_data[2] += shark_state.z;
	h_shark_state[0] = shark_state.x;
	h_shark_state[1] = shark_state.y;
	h_shark_state[2] = shark_state.z;
}

void Renderer::render()
{

	// Enable V-Sync in Code directly
	if ( !shouldUpdate() )
		return;

	prepare();

	shader_.bind();
	va_.bind();

	moveSwarmCenter();															// Set new Swarm center

	Window* window = Window::getInstance();
	currentTime_ = window->getCurrentTime();
	Camera const camera = window->getCamera();

	GLfloat const rotationAngle = static_cast< GLfloat >( 0 ) / 1000.0f * 20.0f;
	glm::mat4x4 const scalingMatrix = glm::scale( glm::mat4( 1.0f ), glm::vec3( 100.0f, 100.0f, 100.0f ) );
	glm::mat4x4 const rotationMatrix = glm::rotate( glm::mat4( 1.0f ), glm::radians( DegreeAngle( rotationAngle ).toFloat() ), glm::vec3( 0.0f, 1.0f, 0.0f ) );
	glm::mat4x4 const modelMatrix = rotationMatrix * scalingMatrix;

	glm::mat4x4 const viewMatrix = camera.viewMatrix();
	glm::mat4x4 const projectionMatrix = camera.projectionMatrix();

	glm::mat4 model( 1.0f );
	glm::mat4 view( 1.0f );
	glm::mat4 projection = glm::ortho( 0.0f, 1.0f * window->getWidth(),
		0.0f, 1.0f * window->getHeight(),
		1.f, -1.f );

	shader_.setUniformMat4f( "u_model", modelMatrix );
	shader_.setUniformMat4f( "u_view", viewMatrix );
	shader_.setUniformMat4f( "u_projection", projectionMatrix );
	shader_.setUniform1f( "u_pointsize", 4.0 );

	
	runCuda();																	// Run Cuda Stuff

	glDrawArrays( GL_POINTS, 0, NUM_PARTICLES );								// Draw particles
	va_.unbind();																// Unbind, because only on VAO can be active.


	/*
	 * Draw Shark
	 */
	vaShark.bind();																// Bind shark VAO
	shader_.setUniform1f("u_pointsize", 15.0);									// Set Point Size bigger than fishies
	moveShark();																// Calculate new shark position on CPU.

	VertexBuffer vbShark(h_shark_data.data(), NUM_SHARKS * 4 * sizeof(float));	// New Vertex Buffer

	VertexBufferLayout layout;													// Layout for new buffer 
	layout.push<float>(4, 0);

	vaShark.addBuffer(vbShark, layout);											// Push new buffer to VAO

	glDrawArrays(GL_POINTS, 0, 1);												// Draw new buffer
	vaShark.unbind();															// Unbind, because only on VAO can be active.
}

void Renderer::cleanUp()
{
	
	device_.unregisterGLBuffer();												// unregister buffer object with CUDA
	
	shader_.unbind();															// Unbind Shader and VAOs
	va_.unbind();
	vaShark.unbind();

	delete d_state;																// Free GPU Memory
	delete d_color;																// Free GPU Memory
}

void Renderer::prepare()
{
	// Set the clear color
	glClearColor( 3.0 / 255.0, 148 / 255.0, 252 / 255.0, 1.0 );					// Set Blue background

	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
}

bool Renderer::shouldUpdate()
{
	Window* window = Window::getInstance();
	double timeDiff = window->getCurrentTime() - getLastUpdate();
	return timeDiff > 0.006;													// Lock to 60 FPS.
}

double Renderer::getLastUpdate()
{
	return lastUpdate_;
}

void Renderer::setLastUpdate( double _lastTime )
{
	lastUpdate_ = _lastTime;
}
