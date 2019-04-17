#ifndef GUARD_dof2dom_h
#define GUARD_dof2dom_h

#include "petsc.h"
#include "type.h"
#include "grid.h"

PetscErrorCode multilayer_forward(PetscScalar *pt, Vec v, DOFInfo *dofi, DM da);

PetscErrorCode multilayer_backward(MPI_Comm comm, Vec v, PetscScalar *pt, DOFInfo *dofi, DM da);

#endif
