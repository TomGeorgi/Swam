#include "waypoint_list.h"

WaypointList::WaypointList()
{}

WaypointList::WaypointList( std::vector<Vector3> waypoints )
{
	append( waypoints );
	idx = head;
}

WaypointList::~WaypointList() 
{
	clear();
}

void WaypointList::append( Vector3 data )
{
	Node* node = new Node();
	Node* last = head;
	
	node->data = data;
	node->next = NULL;
	node->prev = NULL;

	if ( head == NULL )
	{
		head = node;
		head->next = node;
		head->prev = node;
		return;
	}
		
	while ( last->next != NULL && last->next != head )
		last = last->next;

	// Set Next value
	node->next = head;
	last->next = node;

	// Set Prev value
	node->prev = last;
	head->prev = node;
	return;
}

void WaypointList::append( std::vector<Vector3> waypoints )
{
	for ( Vector3 wp : waypoints )
		append( wp );
}

void WaypointList::push( Vector3 data )
{
	Node* node = new Node();

	node->data = data;
	node->next = head;
	node->prev = head->prev;
	head->prev = node;

	head = node;
}

void WaypointList::push( std::vector<Vector3> waypoints )
{
	for (int i = waypoints.size() - 1; i >= 0; i--)
		push( waypoints[i] );
}

void WaypointList::pop()
{
	Node* node = head->next;

	head->next->prev = head->prev;
	head->prev->next = head->next;

	free(head);

	head = node;
}

void WaypointList::remove( int index )
{
	if ( head == NULL )
		return;

	Node* tmp = head;

	if ( index == 0 )
	{
		pop();
		return;
	}

	for ( int i = 0; tmp != head && i < index - 1; i++ )
		tmp = tmp->next;

	if ( tmp->next == head )
		return;

	Node* next = tmp->next->next;

	free( tmp->next );

	tmp->next = next;
	next->prev = tmp;
}

void WaypointList::clear()
{
	Node* current = head;
	Node* next = NULL;

	while ( current != NULL )
	{
		next = current->next;
		free( current );
		current = next;
	}

	head = NULL;
	idx = NULL;
}

int WaypointList::length()
{
	if ( head == NULL )
		return 0;

	int count = 1;
	Node* current = head;
	while ( current->next != head )
	{
		count++;
		current = current->next;
	}

	return count;
}

Vector3 WaypointList::get( int index )
{
	Node* current = head;
	if ( head == NULL )
		return Vector3();

	for ( int i = 0; current->next != head && i < index; i++ )
		current = current->next;

	if ( current->next == head )
		return Vector3();

	return current->data;
}

Vector3 WaypointList::get()
{
	if ( idx == NULL )
		idx = head;
	return idx->data;
}

Vector3 WaypointList::getNext()
{
	if ( idx == NULL )
		idx = head;
	else
		idx = idx->next;
	return idx->data;
}

Vector3 WaypointList::getPrev()
{
	if ( idx == NULL )
		idx = head;
	else
		idx = idx->prev;
	return idx->data;
}