#ifndef PARTICLE_H
#define PARTICLE_H

#include "Common.h"
#include "matrix.h"

struct Particle {
	float3 pos;
	float3 velocity;
	float mass;
	float volume;
	mat3 fe;
	mat3 fp;

	Particle(float3 pos, float3 velocity, float mass) :
		pos(pos), velocity(velocity), mass(mass), volume(0),
		fe(mat3(1.0f)), fp(mat3(1.0f))
	{}
};

#endif