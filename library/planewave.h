#ifndef GUARD_planewave_h
#define GUARD_planewave_h

#include "petsc.h"
#include "type.h"
#include "ovmat.h"
#include "array2vec.h"

void planewave(MPI_Comm big_comm, int colour, int ncells_per_comm, PetscScalar **vx, PetscScalar **vy, int nx, int ny, int px, int py, int numcells_x, int numcells_y, PetscScalar val_margin, double dx, double dy, double kx, double ky, PetscScalar ax, PetscScalar ay, PetscBool conjugate);

#endif
