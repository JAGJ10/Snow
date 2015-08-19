#include "ParticleSystem.h"
#include "Simulation.cuh"

using namespace std;

ParticleSystem::ParticleSystem(vector<Particle>& particles, solverParams& params) {

	solverParams sp = params;

	cout << "Solver parameters:" << endl;
	cout << "alpha: " << sp.alpha << endl;
	cout << "boxCorner1: " << "(" << sp.boxCorner1.x << ", " << sp.boxCorner1.y << ", " << sp.boxCorner1.z << ")" << endl;
	cout << "boxCorner2: " << "(" << sp.boxCorner2.x << ", " << sp.boxCorner2.y << ", " << sp.boxCorner2.z << ")" << endl;
	cout << "compression: " << sp.compression << endl;
	cout << "dt: " << sp.deltaT << endl;
	cout << "density: " << sp.density << endl;
	cout << "frictionCoeff: " << sp.frictionCoeff << endl;
	cout << "gBounds: " << "(" << sp.gBounds.x << ", " << sp.gBounds.y << ", " << sp.gBounds.z << ")" << endl;
	cout << "gravity: " << "(" << sp.gravity.x << ", " << sp.gravity.y << ", " << sp.gravity.z << ")" << endl;
	cout << "hardening: " << sp.hardening << endl;
	cout << "lambda: " << sp.lambda << endl;
	cout << "mu: " << sp.mu << endl;
	cout << "numParticles: " << sp.numParticles << endl;
	cout << "poisson: " << sp.poisson << endl;
	cout << "radius: " << sp.radius << endl;
	cout << "stickyWalls: " << sp.stickyWalls << endl;
	cout << "stretch: " << sp.stretch << endl;
	cout << "young: " << sp.young << endl;

	cudaCheck(cudaMalloc((void**)&this->particles, params.numParticles * sizeof(Particle)));
	cudaCheck(cudaMalloc((void**)&cells, params.gridSize * sizeof(Cell)));

	//Copy data to device
	cudaCheck(cudaMemcpy(this->particles, &particles[0], params.numParticles * sizeof(Particle), cudaMemcpyHostToDevice));
}

ParticleSystem::~ParticleSystem() {
	cudaCheck(cudaFree(particles));
	cudaCheck(cudaFree(cells));
}

void ParticleSystem::updateWrapper(solverParams& params) {
	setParams(&params);
	update(particles, cells, params.gridSize);
}

void ParticleSystem::getPositionsWrapper(float* positionsPtr) {
	getPositions(positionsPtr, particles);
}