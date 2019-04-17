#ifndef GUARD_array2vec_h
#define GUARD_array2vec_h

#include "petsc.h"
#include "type.h"

PetscErrorCode array2mpi_c2c(PetscScalar *pt, Vec v);

PetscErrorCode array2mpi_f2c(PetscReal *pt, Vec v);

PetscErrorCode mpi2array_c2c(Vec v, PetscScalar *pt, int n);

PetscErrorCode mpi2array_c2f(Vec v, PetscReal *pt, int n);

#endif
