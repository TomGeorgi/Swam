#pragma once
#include <iostream>
#include <vector>

#include "vec3.h"

/*!
 * @brief WaypointList contains waypoints for particles.
 * Contains methods to handle linked list.
 */
class WaypointList
{
private:

	/*!
	 * @brief Node for linked list.
	 */
	class Node
	{
	public:
		Vector3 data;	//!< Waypoint
		Node* next;		//!< Next node in linked list
		Node* prev;		//!< Previous node in linked list
	};

	Node* head = NULL;	//!< Head node of linked list
	Node* idx = NULL;	//!< Current index of linked list. Used for getter methods.

public:
	/*!
	 * @brief Default Constructor.
	 */
	WaypointList();

	/*!
	 * @brief Constructor initializes linked list with given waypoints
	 * @param waypoints waypoints for particles.
	 */
	WaypointList( std::vector<Vector3> waypoints );

	/*!
	 * @brief Default Destructor. Destroys linked list.
	 */
	~WaypointList();

	/*!
	 * @brief Append a new waypoint
	 * @param data waypoint
	 */
	void append( Vector3 data );

	/*!
	 * @brief Append a list of waypoints
	 * @param waypoints list of waypoints
	 */
	void append( std::vector<Vector3> waypoints );

	/*!
	 * @brief push waypoint as first item in linked list.
	 * @param data waypoint.
	 */
	void push( Vector3 data );

	/*!
	 * @brief push waypoints as first items in linked list.
	 * @param waypoints list of waypoints
	 */
	void push( std::vector<Vector3> waypoints );

	/*!
	 * @brief pop first waypoint in linked list
	 */
	void pop();

	/*!
	 * @brief remove waypoint at the given index
	 * @param index waypoint index.
	 */
	void remove( int index );

	/*!
	 * @brief clear waypoints.
	 */
	void clear();

	/*!
	 * @brief get number of waypoints.
	 * @return length
	 */
	int length();

	/*!
	 * @brief get waypoint at the given index.
	 * @param index index
	 * @return v(0, 0, 0, 1) if head is empty or index is out of scope or vector at the given valid position.
	 */
	Vector3 get( int index );

	/*!
	 * @brief get current waypoint
	 * @return current waypoint
	 */
	Vector3 get();

	/*!
	 * @brief Get next waypoint in list.
	 * @return Next waypoint
	 */
	Vector3 getNext();

	/*!
	 * @brief Get previous waypoint in list.
	 * @return Previous waypoint.
	 */
	Vector3 getPrev();

};