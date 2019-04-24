#ifndef GUARD_phprofile_h
#define GUARD_phprofile_h

#include "petsc.h"
#include "type.h"
#include "ovmat.h"
#include "array2vec.h"

void phdiff(MPI_Comm big_comm, int colour, int ncells_per_comm, PetscScalar **vx, PetscScalar **vy, int nx, int ny, int px, int py, int numcells_x, int numcells_y, double dx, double dy, double freq, double oxy[2], double xyzfar[3], double axy[2], PetscBool conjugate);

#endif
