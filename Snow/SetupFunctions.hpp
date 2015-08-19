#ifndef SETUP_FUNCTIONS_H
#define SETUP_FUNCTIONS_H

#include "Common.h"
#include "Parameters.h"
#include "Particle.h"
#include <time.h>

inline void createParticleGrid(std::vector<Particle>& particles, solverParams* sp, float3 lower, int3 dims, float radius, float mass) {
	//srand(int(time(NULL)));
	srand(16);

	for (int x = 0; x < dims.x; x++) {
		for (int y = 0; y < dims.y; y++) {
			for (int z = 0; z < dims.z; z++) {
				float r1 = 0.001f + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				float r2 = 0.001f + static_cast <float>(rand()) / static_cast <float> (RAND_MAX);
				float r3 = 0.001f + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				float3 jitter = make_float3(r1, r2, r3) * radius;
				float3 pos = lower + make_float3(float(x), float(y), float(z)) * radius + jitter;
				float3 velocity = make_float3(0);
				particles.push_back(Particle(pos, velocity, mass));
			}
		}
	}
}

//Some method for creating a snowball
inline void createSnowball(std::vector<Particle>& particles, float3 center, int3 dims, float radius, float mass, float3 velocity) {
	float sphereRadius = radius * (float)dims.x / 2.0f;
	for (int x = -dims.x/2; x <= dims.x/2; x++) {
		for (int y = -dims.y/2; y <= dims.y/2; y++) {
			for (int z = -dims.z/2; z <= dims.z/2; z++) {
				// generate a jittered point
				float r1 = 0.001f + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				float r2 = 0.001f + static_cast <float>(rand()) / static_cast <float> (RAND_MAX);
				float r3 = 0.001f + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				float3 jitter = make_float3(r1, r2, r3) * radius;

				float3 pos = center + make_float3(float(x), float(y), float(z)) * radius + jitter;
				// see if pos is inside the sphere
				if (length(pos - center) < sphereRadius) {
					particles.push_back(Particle(make_float3(pos.x, pos.y, pos.z), make_float3(velocity.x, velocity.y, velocity.z), mass));
				}
			}
		}
	}


}


#endif