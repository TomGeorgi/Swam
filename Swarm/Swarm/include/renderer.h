#pragma once

#include "cuda_device.h"
#include "cuda_device_array.h"
#include "shader.h"
#include "vertex_array.h"
#include "waypoint_list.h"

/*!
 * @brief Renderer is used as main class.
 * Class contains methods to render the scene.
 */
class Renderer
{
private:

	Shader shader_;							//!< Contains Shader (Vertex und Fragment shader).
	VertexArray va_;						//!< Vertex Array to render particles.
	VertexArray vaShark;					//!< Vertex Array to render shark.
	
	CudaDevice device_;						//!< Cuda Device. Used to simply communicate with the gpu.

	std::vector<float> h_data;				//!< contains positions on host.
	std::vector<float> h_color;				//!< contains colors on host.
	std::vector<float> h_state;				//!< contains vec3 force and mass on host.
	std::vector<float> h_shark_data;		//!< contains shark position on host.
	std::vector<float> h_shark_color;		//!< contains shark color on host.
	std::vector<float> h_shark_state;		//!< contains force and mass on host.

	CudaDeviceArray<float>* d_state;		//!< contains force and mass in memory on device.
	CudaDeviceArray<float>* d_color;		//!< contains color in memory on device.

	const float WAYPOINT_THRESHOLD = 0.1;	//!< Threshold for waypoint goal.

	float speed = 0.015;					//!< speed of particles.
	WaypointList* waypointList;				//!< Waypoint List contains waypoints for particles.
	Vector3 swarmCenter;					//!< Position of swarm center.

	double lastUpdate_, currentTime_;		//!< times for v-sync.

	/*!
	 * @brief Create Buffers.
	 * Allocate memory on GPU and initialize gpu grid.
	 * Also register VBO with CUDA.
	 */
	void createBuffers();

	/*!
	 * @brief Call CUDA Function to calculate new positions per particle.
	 */
	void runCuda();

	/*!
	 * @brief Move Swarm center to waypoint
	 */
	void moveSwarmCenter();

	/*!
	 * @brief Move shark in a pseudo realistic manner.
	 * It moves roughly through the swarm center to maximise probability of catching a fish.
	 * Sometimes circles around the swarm.
	 */
	void moveShark();


public:
	static const unsigned int NUM_PARTICLES = 1000;	//!< Number of Particles
	static const unsigned int NUM_SHARKS = 1;		//!< Number of Sharks. While on CPU only one shark.

	/*!
	 * @brief Default Constructor. 
	 * Initialize Waypoints, Buffers and Shader.
	 */
	Renderer();

	/*!
	 * @brief Render new scene.
	*/
	void render();

	/*!
	 * @brief Free Memory on GPU. Unbind Shader and VAOs.
	 */
	void cleanUp();

private:

	/*!
	 * @brief Preparing Scene on new rendering.
	 */
	void prepare();

	/*!
	 * @brief Check if scene should be updated.
	 * @return Either true or false.
	 */
	bool shouldUpdate();

	/*!
	 * @brief Get Time of Last update
	 * @return time of last update as float
	 */
	double getLastUpdate();

	/*!
	 * @brief Set time of last update.
	 * @param _lastTime time of last update.
	 */
	void setLastUpdate(double _lastTime);
};
