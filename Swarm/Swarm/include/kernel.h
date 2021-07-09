#pragma once
#include "vec3.h"
#include "renderer.h"

using namespace std;

/*!
 * @brief Call Kernel to calculate new positions.
 * @param verts Vertices
 * @param states States of particles
 * @param mesh_count Number of particles
 * @param speed speed of particles
 * @param swarmCenter swarm center
 * @param shark shark position
*/
void kernel_advance(
    float4* verts,
    float4* states,
    unsigned int mesh_count,
    float speed,
    Vector3 swarmCenter,
    Vector3 shark);

/*!
 * @brief Initialization of kernel related values.
 * @param mesh_count Number of fishies.
*/
void kernel_init_grid(int mesh_count);
