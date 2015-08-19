#ifndef SIMULATION_CUH
#define SIMULATION_CUH

#include "Parameters.h"
#include "Particle.h"
#include "Cell.h"

void update(Particle* particles, Cell* cells, int gridSize);
void getPositions(float* positionsPtr, Particle* particles);
void setParams(solverParams *tempParams);

#endif