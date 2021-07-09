#include <cuda.h>
#include <cuda_runtime.h>
#include <kernel.h>
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

static int BLOCK_SIZE = 128;
static int NUM_BLOCKS;
static int NUM_THREADS;
static int NUM_BLOCKS_DISTANCE;

__device__ float CENTER_THRESHOLD = 2.0;
__device__ float SHARK_DIST = 0.7;
__device__ float SHARK_BITE_DIST = 0.05;
__device__ float FISH_DIST = 0.4;
__device__ float ACCELERATION_FACTOR = 0.2;

class DeviceVector;

/********************************
 *
 * DeviceVector class definition
 *
 ********************************/

// Must be referenced here
// Because the CUDA Compiler can't access device functions
// in other context (*.cu files)

/*!
 * @brief Class to make vector math easier inside the kernel.
*/
class DeviceVector
{
public:
	float x;
	float y;
	float z;
	float w;

public:

	/*!
	 * @brief Default Constructor.
	 */
	__host__ __device__ DeviceVector() :
		x( 0 ), y( 0 ), z( 0 ), w( 1 )
	{}

	/*!
	 * @brief Copy Constructor.
	 * @param v DeviceVector instance.
	 */
	__host__ __device__ DeviceVector( DeviceVector* v ) :
		x( v->x ), y( v->y ), z( v->z ), w( v->w )
	{}

	/*!
	 * @brief Copy Constructor.
	 * @param v Vector3 instance.
	 */
	__host__ __device__ DeviceVector( Vector3* v ) :
		x( v->x ), y( v->y ), z( v->z ), w( 1 )
	{}

	/*!
	 * @brief Constructor.
	 * @param v float4 struct.
	 */
	__host__ __device__ DeviceVector( float4 v ) :
		x( v.x ), y( v.y ), z( v.z ), w( v.w )
	{}

	/*!
	 * @brief Constructor.
	 * @param x x value.
	 * @param y y value.
	 * @param z z value.
	 */
	__host__ __device__ DeviceVector( float x, float y, float z ) :
		x( x ), y( y ), z( z ), w( 1 )
	{}

	/*!
	 * @brief Constructor.
	 * @param x x value.
	 * @param y y value.
	 * @param z z value.
	 * @param w w value.
	 */
	__host__ __device__ DeviceVector( float x, float y, float z, float w ) :
		x( x ), y( y ), z( z ), w( w )
	{}

	/*!
	 * @brief Calculate Dot Product of two Device Vectors.
	 * @param vec Other DeviceVector instance.
	 * @return result of dot product.
	 */
	__device__ float dot( DeviceVector* vec )
	{
		return x * vec->x + y * vec->y + z * vec->z;
	}

	/*!
	 * @brief Get Length of DeviceVector.
	 * @return length
	 */
	__device__ float length()
	{
		return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(w, 2) );
	}

	/*!
	 * @brief Get Length of DeviceVector(3)
	 * @return length
	 */
	__device__ float length3()
	{
		return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
	}

	/*!
	 * @brief Fill float4 struct with values of DeviceVector.
	 * @param fill float4 struct pointer
	 */
	__device__  void getFloat4( float4* fill )
	{
		fill->x = x;
		fill->y = y;
		fill->z = z;
		fill->w = w;
	}

	/*!
	 * @brief fill float4 struct with vector values.
	 * @return float4 struct
	 */
	__device__  float4 getFloat4()
	{
		float4 v;
		v.x = x;
		v.y = y;
		v.z = z;
		v.w = w;
		return v;
	}

	/*!
	 * @brief normalize vector.
	 * @return Normalized DeviceVector.
	 */
	__device__ DeviceVector normalized()
	{
		float len = length();
		return DeviceVector( x / len, y / len, z / len );
	}

	/*!
	 * @brief Add two vectors.
	 * @param vec other vector instance.
	 * @return sum of devicevectors.
	 */
	__device__ DeviceVector operator+( const DeviceVector& vec )
	{
		return DeviceVector( x + vec.x, y + vec.y, z + vec.z );
	}

	/*!
	 * @brief sum up device vectors.
	 * @param vec other vector instance.
	 */
	__device__ void operator+=( const DeviceVector& vec )
	{
		x += vec.x;
		y += vec.y;
		z += vec.z;
	}

	/*!
	 * @brief Subtract two vectors.
	 * @param vec other vector instance.
	 * @return Difference of devicevectors.
	 */
	__device__ DeviceVector operator-( const DeviceVector& vec )
	{
		return DeviceVector( x - vec.x, y - vec.y, z - vec.z );
	}

	/*!
	 * @brief Difference of two device vectors.
	 * @param vec other vector instance.
	 */
	__device__ void operator-=(const DeviceVector& vec)
	{
		x -= vec.x;
		y -= vec.y;
		z -= vec.z;
	}

	/*!
	 * @brief Multiply two vectors.
	 * @param vec other vector instance.
	 * @return Product of devicevectors.
	 */
	__device__ DeviceVector operator*( const DeviceVector& vec )
	{
		return DeviceVector( x * vec.x, y * vec.y, z * vec.z );
	}

	/*!
	 * @brief Product of device vector with float.
	 * @param number float value.
	 * @return new DeviceVector result.
	 */
	__device__ DeviceVector operator*( const float number )
	{
		return DeviceVector( x * number, y * number, z * number );
	}


	/*!
	 * @brief Product of device vector with float.
	 * @param number float value.
	 */
	__device__ void operator*=(const float number)
	{
		x *= number;
		y *= number;
		z *= number;
	}
};


/********************************
 *
 * Kernel Stuff
 *
 ********************************/


/*
 * The following 2 functions taken from the cuda samples
 */
int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

/**
 * Compute the ideal grid size for the number of particles that we have
 */
void computeGridSize(int n, int blockSize)
{
    NUM_THREADS = std::min(blockSize, n);
    NUM_BLOCKS = iDivUp(n, NUM_THREADS);
}

/*!
 * @brief Kernel function that controls behavior of a fish in the swarm.
 * Fishies can be eaten by shark, try to evade shark, keep distance to other fishies and return to swarm when to far away.
 * At the moment not effiecient due to big for loop that iterates over all fishies in order to get the closest.
 * Swarm behavior can be modified by changing global variables at the beginning of this file.
 * @param verts Positions of all fishies.
 * @param states Speed Vectors of all fishies (x, y z) and a random variable (w) to avoid creating new random values on the GPU all the time.
 * @param mesh_count Number of fishies.
 * @param speed Approximate maximum speed of fishies. Approximate because it can get higher depending on random value states.w.
 * @param swarmCenter The center of the swarm that each fish tries to reach.
 * @param sharkVec Position of the shark.
*/
__global__ void d_advance(
    float4* verts,
    float4* states,
    unsigned int mesh_count,
    float speed,
    Vector3 swarmCenter,
	Vector3 sharkVec)
{
    int t_x = threadIdx.x;
    int b_x = blockIdx.x;
    int in_x = b_x * blockDim.x + t_x;

    DeviceVector vert(verts[in_x]);
	DeviceVector state(states[in_x]);
	DeviceVector shark(&sharkVec);
	float my_speed = speed * state.w;
	float acceleration_factor = 0.09;

	DeviceVector sharkDiff = shark - vert;
	float sharkDistance = sharkDiff.length3();

	// shark eats fish
	if (sharkDistance < SHARK_BITE_DIST || state.w < 0)
	{
		verts[in_x].w = -1;
		return;
	}
	// evade shark
	if (sharkDistance < SHARK_DIST * state.w)
	{
		sharkDiff = sharkDiff.normalized() * my_speed * acceleration_factor;
		state -= sharkDiff;
	}
	else
	{
		DeviceVector center(&swarmCenter);
		DeviceVector closest = vert - DeviceVector(verts[0]);
		float closest_dist = closest.length3();

		// find closest fish
		DeviceVector d;
		float d_len;
		for (int i = 1; i < mesh_count; i++)
		{ 
			d = vert - verts[i];
			d_len = d.length3();
			if ( d_len < closest_dist && i != in_x)
			{
				closest = d;
				closest_dist = d_len;
			}
		}

		DeviceVector diff = center - vert;

		// keep distance to other fishies
		bool too_close = closest_dist < FISH_DIST;
		if (too_close)
		{
			DeviceVector avoid = closest.normalized() * my_speed * acceleration_factor * 0.7;
			state -= avoid;
			vert += state;
			acceleration_factor /= 2;
		}
		// return to swarm
		if (diff.length3() > CENTER_THRESHOLD * state.w)
		{
			diff = diff.normalized() * my_speed * (acceleration_factor * 0.4);
			state += diff;
		}
	}
	if (state.length3() > my_speed * 0.75)
	{
		state *= 0.96;
	}
	vert += state;

	float4 result;
	vert.getFloat4(&result);
	verts[in_x] = result;
	float4 state_result;
	state.getFloat4(&state_result);
	states[in_x] = state_result;
}

void kernel_advance(
    float4* verts,
    float4* states,
	unsigned int mesh_count,
    float speed,
    Vector3 swarmCenter,
	Vector3 shark)
{
	// KERNEL CALL
    d_advance<<<NUM_BLOCKS, NUM_THREADS>>> ( verts, states, mesh_count, speed * 1.8, swarmCenter, shark );
}

void kernel_init_grid(int mesh_count)
{
	// Compute optimal grid size depending on number 
	// of fishies and pre defined block size.
    computeGridSize(mesh_count, BLOCK_SIZE);
}


