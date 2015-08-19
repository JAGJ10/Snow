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
		sp->deltaT = 1e-5f;
		sp->radius = 0.05f;
		sp->compression = 0.019f;
		sp->stretch = 0.0075f;
		sp->hardening = 25.0f;
		sp->young = 4.8e4f;
		sp->poisson = 0.2f;
		sp->alpha = 0.95f;
		sp->density = 50.0;

		sp->lambda = getLambda(sp->poisson, sp->young);
		sp->mu = getMu(sp->poisson, sp->young);

		sp->gravity = make_float3(0, -9.8f, 0);

		sp->frictionCoeff = .5f;
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

		const float restDistance = sp->radius * 0.25f;
		float3 lower = make_float3(0.5f, 0.5f, 0.5f);
		int3 dims = make_int3(25, 25, 25);
		createParticleGrid(particles, sp, lower, dims, restDistance, getMass(sp->radius, sp->density));

		sp->numParticles = int(particles.size());
		sp->gridSize = dims.x * dims.y * dims.z;
		sp->gBounds = dims;

		// TODO: better bounds and set plane to respect bottom y
		sp->boxCorner1 = make_float3(0, 0, 0);
		sp->boxCorner2 = make_float3(dims.x * sp->radius, dims.y * sp->radius, dims.z * sp->radius);
	}
};

class SingleParticle : public Scene {
public:
	SingleParticle(std::string name) : Scene(name) {}

	virtual void init(std::vector<Particle>& particles, solverParams* sp) {

		Scene::init(particles, sp);

		const float radius = 0.1f;
		const float restDistance = radius * 0.5f;
		float3 lower = make_float3(0, 1.0f, 0);
		int3 dims = make_int3(25, 25, 25);

		float3 pos = make_float3(0.0f, 0.25f, 0.0f);
		float3 velocity = make_float3(0);
		float mass = getMass(sp->radius, sp->density);
		particles.push_back(Particle(pos, velocity, mass));

		sp->numParticles = int(particles.size());
		sp->deltaT = 1e-3f;
		sp->gridSize = dims.x * dims.y * dims.z;
		sp->gBounds = dims;
	}
};

class SnowballDrop : public Scene {
public:
	SnowballDrop(std::string name) : Scene(name) {}

	virtual void init(std::vector<Particle>& particles, solverParams* sp) {
		Scene::init(particles, sp);
		sp->radius = 0.017f;
		const float restDistance = sp->radius * 0.25f;

		int3 dims = make_int3(150);

		sp->boxCorner1 = make_float3(0, .1f, 0);
		sp->boxCorner2 = make_float3((dims.x) * sp->radius, (dims.y) * sp->radius, (dims.z) * sp->radius);

		// seed the snowmall with a bit of sideways and downward velocity, as if it's been dropping for a while
		// radius of sphere is .3125
		float3 snowDims = make_float3(30);
		createSnowball(particles, make_float3(1.25f, 1.8f, 1.25f), make_int3(dims.x, dims.y, dims.z), restDistance, getMass(sp->radius, sp->density), make_float3(.2f, -1, 0));

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
		sp->radius = 0.017f;
		const float restDistance = sp->radius * 1.5f;

		int3 dims = make_int3(150);

		sp->boxCorner1 = make_float3(0.2f, .1f, 0);
		sp->boxCorner2 = make_float3((dims.x) * sp->radius, (dims.y) * sp->radius, (dims.z) * sp->radius);

		// seed the snowmall with a bit of sideways and downward velocity, as if it's been dropping for a while
		// radius of sphere is .3125
		int3 snowDims = make_int3(30);
		createSnowball(particles, make_float3(1.0f, 1.0f, 1.25f), snowDims, restDistance, getMass(sp->radius, sp->density), make_float3(-15.0f, 0.0f, 0));

		sp->numParticles = int(particles.size());
		sp->gBounds = dims;
		sp->gridSize = dims.x * dims.y * dims.z;
		sp->stickyWalls = true;
	}
};

#endif