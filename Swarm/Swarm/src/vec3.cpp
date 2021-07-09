#include "vec3.h"

Vector3::Vector3()
{
	x = NULL; y = NULL; z = NULL;
}

Vector3::Vector3( float x, float y, float z ) : x(x), y(y), z(z)
{}

float Vector3::dot( Vector3 * vec )
{
	return x * vec->x + y * vec->y + z * vec->z;
}

float Vector3::length()
{
	return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
}

Vector3 Vector3::normalized()
{
	float len = length() + 1e-10;
	return Vector3(x / len, y / len, z / len);
}
