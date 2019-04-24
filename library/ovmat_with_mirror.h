#ifndef GUARD_ovmat_with_mirror_h
#define GUARD_ovmat_with_mirror_h

#include "petsc.h"
#include "type.h"

void ovmatsym(MPI_Comm comm, Mat *Wout, int nx, int ny, int px, int py, int numcells_x, int numcells_y, int numlayers, PetscScalar val_margin, double kLx, double kLy, int mirrorXY[2]);

#endif
