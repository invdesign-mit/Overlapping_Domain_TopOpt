#ifndef GUARD_near2far_h
#define GUARD_near2far_h

#include "petsc.h"
#include "type.h"
#include "ovmat.h"
#include "array2vec.h"

void green3d(PetscScalar *EH, const double *x,
	     double freq, double eps, double mu,
	     const double *x0, int comp, PetscScalar f0);

void create_near2far(MPI_Comm big_comm, int colour, int ncells_per_comm,
		     PetscScalar **ux, PetscScalar **uy,
		     PetscScalar **vx, PetscScalar **vy,
		     PetscScalar **wx, PetscScalar **wy,
		     int nx, int ny, int px, int py, int numcells_x, int numcells_y, double dx, double dy,
		     double oxy[2], int symxy[2],
		     double xyzfar[3],
		     double freq, double eps, double mu, PetscBool conjugate);

#endif
