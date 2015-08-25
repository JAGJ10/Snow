#ifndef SCENE_H
#define SCENE_H

#include "Common.h"
#include "Parameters.h"
#include "SetupFunctions.hpp"
#include "Particle.h"

class Scene {
public:
	Scene(std::string name) : name(name) {}
	virtual void init(std::vector<Particle>& particles, solverParams* sp) {
		sp->deltaT = 5e-5f;
		sp->radius = 0.017f;
		sp->compression = 0.019f;
		sp->stretch = 0.0075f;
		sp->hardening = 15.0f;
		sp->young = 4.8e4f;
		sp->poisson = 0.2f;
		sp->alpha = 0.95f;
		sp->density = 100.0f;

		sp->lambda = getLambda(sp->poisson, sp->young);
		sp->mu = getMu(sp->poisson, sp->young);

		sp->gravity = make_float3(0, -9.8f, 0);

		sp->frictionCoeff = 1.0f;
		sp->stickyWalls = false;
	}

	float getLambda(float poisson, float young) {
		return (poisson * young) / ((1 + poisson) * (1 - 2 * poisson));
	}

	float getMu(float poisson, float young) {
		return young / (2 * (1 + poisson));
	}

	float getMass(float radius, float density) {
		return pow(radius, 3) * density / 4;
	}

private:
	std::string name;
};

class GroundSmash : public Scene {
public:
	GroundSmash(std::string name) : Scene(name) {}

	virtual void init(std::vector<Particle>& particles, solverParams* sp) {
		Scene::init(particles, sp);

		const float restDistance = sp->radius * 1.f;
		
		int3 dims = make_int3(150);
		int3 snowDims = make_int3(40);

		sp->boxCorner1 = make_float3(0, 0.0f, 0);
		sp->boxCorner2 = make_float3((dims.x) * sp->radius, (dims.y) * sp->radius, (dims.z) * sp->radius);

		float3 lower = make_float3(dims.x / 2 * sp->radius, 0.5f, dims.z / 2 * sp->radius);
		createParticleGrid(particles, sp, lower, snowDims, restDistance, getMass(sp->radius, sp->density), make_float3(0, -5, 0));

		sp->numParticles = int(particles.size());
		sp->gridSize = dims.x * dims.y * dims.z;
		sp->gBounds = dims;
	}
};

class SnowballDrop : public Scene {
public:
	SnowballDrop(std::string name) : Scene(name) {}

	virtual void init(std::vector<Particle>& particles, solverParams* sp) {
		Scene::init(particles, sp);
		const float restDistance = sp->radius * 1.0f;

		int3 dims = make_int3(150);

		sp->boxCorner1 = make_float3(0, 0.0f, 0);
		sp->boxCorner2 = make_float3((dims.x) * sp->radius, (dims.y) * sp->radius, (dims.z) * sp->radius);

		int3 snowDims = make_int3(30);
		createSnowball(particles, make_float3(1.25f, 1.0f, 1.25f), snowDims, restDistance, getMass(sp->radius, sp->density), make_float3(0, -10, 0));

		sp->numParticles = int(particles.size());
		sp->gBounds = dims;
		sp->gridSize = dims.x * dims.y * dims.z;
	}
};


class WallSmash : public Scene {
public:
	WallSmash(std::string name) : Scene(name) {}

	virtual void init(std::vector<Particle>& particles, solverParams* sp) {
		Scene::init(particles, sp);
		const float restDistance = sp->radius * 1.0f;

		int3 dims = make_int3(150);

		sp->boxCorner1 = make_float3(0.0f, 0.0f, 0);
		sp->boxCorner2 = make_float3((dims.x) * sp->radius, (dims.y) * sp->radius, (dims.z) * sp->radius);

		int3 snowDims = make_int3(40);
		createSnowball(particles, make_float3(1.0f, 1.0f, 1.25f), snowDims, restDistance, getMass(sp->radius, sp->density), make_float3(-10.0f, 0.0f, 0));

		sp->numParticles = int(particles.size());
		sp->gBounds = dims;
		sp->gridSize = dims.x * dims.y * dims.z;
		sp->stickyWalls = true;
	}
};

#endif