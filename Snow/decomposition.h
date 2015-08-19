/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   decomposition.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 13 Apr 2014
**
**************************************************************************/

#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "matrix.h"
#include "helper_math.h"

#include "svd3/svd3_cuda/svd3_cuda.h"

__host__ __device__ __forceinline__ void computeSVD(const mat3 &A, mat3 &W, mat3 &S, mat3 &V) {

	svd(A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8],
		W[0], W[3], W[6], W[1], W[4], W[7], W[2], W[5], W[8],
		S[0], S[3], W[6], S[1], S[4], S[7], S[2], S[5], S[8],
		V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
}

/*
* Returns polar decomposition of 3x3 matrix M where
* M = Fe = Re * Se = U * P
* U is an orthonormal matrix
* S is symmetric positive semidefinite
* Can get Polar Decomposition from SVD, see first section of http://en.wikipedia.org/wiki/Polar_decomposition
*/
__host__ __device__ void computePD(const mat3 &A, mat3 &R) {
	// U is unitary matrix (i.e. orthogonal/orthonormal)
	// P is positive semidefinite Hermitian matrix
	mat3 W, S, V;
	computeSVD(A, W, S, V);
	R = mat3::multiplyABt(W, V);
}

/*
* Returns polar decomposition of 3x3 matrix M where
* M = Fe = Re * Se = U * P
* U is an orthonormal matrix
* S is symmetric positive semidefinite
* Can get Polar Decomposition from SVD, see first section of http://en.wikipedia.org/wiki/Polar_decomposition
*/
__host__ __device__ void computePD(const mat3 &A, mat3 &R, mat3 &P) {
	// U is unitary matrix (i.e. orthogonal/orthonormal)
	// P is positive semidefinite Hermitian matrix
	mat3 W, S, V;
	computeSVD(A, W, S, V);
	R = mat3::multiplyABt(W, V);
	P = mat3::multiplyADBt(V, S, V);
}

/*
* In snow we desire both SVD and polar decompositions simultaneously without
* re-computing USV for polar.
* here is a function that returns all the relevant values
* SVD : A = W * S * V'
* PD : A = R * E
*/
__host__ __device__ void computeSVDandPD(const mat3 &A, mat3 &W, mat3 &S, mat3 &V, mat3 &R) {
	computeSVD(A, W, S, V);
	R = mat3::multiplyABt(W, V);
}

#endif // DECOMPOSITION_H
