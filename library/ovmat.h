#ifndef GUARD_ovmat_h
#define GUARD_ovmat_h

#include "petsc.h"
#include "type.h"

void ovmat(MPI_Comm comm, Mat *Wout, int nx, int ny, int px, int py, int numcells_x, int numcells_y, int numlayers, PetscScalar val_margin, double kLx, double kLy);

PetscErrorCode vecfill_zslice(MPI_Comm comm, DM da, int mx, int my, PetscScalar *vx, PetscScalar *vy, PetscScalar *vz, Vec v, int iz0);

#endif
