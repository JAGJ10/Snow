#ifndef CELL_H
#define CELL_H

#include "Common.h"

struct Cell {
	float mass;
	float3 force;
	float3 velocity;
	float3 velocityStar; //predicted velocity

	Cell() : mass(0.0f), force(make_float3(0.0f)), velocity(make_float3(0.0f)), velocityStar(make_float3(0.0f)) {}
};

#endif