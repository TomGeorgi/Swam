#pragma once

#include <iostream>
#include <cmath>

using namespace std;

/*!
 * @brief Vector3 Class to modify a vector.
 */
class Vector3
{
public:
	float x;	//!< x value
	float y;	//!< y value
	float z;	//!< z value

public:

	/*!
	 * @brief Standard Constructor.
	 */
	Vector3();

	/*!
	 * @brief Constructor to set a vector.
	 * @param x x value.
	 * @param y y value.
	 * @param z z value.
	 */
	Vector3( float x, float y, float z );

	/*!
	 * @brief Multiplicate to vectors.
	 * @param vec an other vector.
	 * @return Get Euclidian distance.
	 */
	float dot( Vector3* vec );

	/*!
	 * @brief Get length of the vector.
	 * @return length.
	 */
	float length();

	/*!
	 * @brief Norm Vector.
	 * @return a normalized vector.
	 */
	Vector3 normalized();

	/*!
	 * @brief add two vectors.
	 * @param vec other vector.
	 * @return sum of this vectors in a new vector.
	 */
	Vector3 operator+( const Vector3& vec )
	{
		return Vector3( x + vec.x, y + vec.y, z + vec.z );
	}

	/*!
	 * @brief Sum up vectors.
	 * @param vec add a vector to the instance.
	 */
	void operator+=(const Vector3& vec)
	{
		x += vec.x;
		y += vec.y;
		z += vec.z;
	}

	/*!
	 * @brief subtract two vectors.
	 * @param vec other vector.
	 * @return difference of this vectors in a new vector.
	 */
	Vector3 operator-( const Vector3& vec )
	{
		return Vector3( x - vec.x, y - vec.y, z - vec.z );
	}

	/*!
	 * @brief multiply two vectors.
	 * @param vec other vector.
	 * @return product of this vectors in a new vector.
	 */
	Vector3 operator*( const Vector3& vec )
	{
		return Vector3( x * vec.x, y * vec.y, z * vec.z );
	}

	/*!
	 * @brief multiply vector with a number.
	 * @param number number.
	 * @return product of this vectors in a new vector.
	 */
	Vector3 operator*(const float number)
	{
		return Vector3(x * number, y * number, z * number);
	}

	/*!
	 * @brief Print out a Vector3 instance with iostream.
	 * @param os stream.
	 * @param v vector.
	 * @return vector stream.
	 */
	friend ostream& operator<<( ostream& os, const Vector3& v )
	{
		os << "X: " << v.x << ", Y: " << v.y << ", Z: " << v.z;
		return os;
	}
};