#ifndef GUARD_output_h
#define GUARD_output_h

#include "petsc.h"
#include "hdf5.h"
#include "petscviewerhdf5.h" 

void writetofile_c2c(MPI_Comm comm, char *name, PetscScalar *data, PetscInt n);

void writetofile_c2f(MPI_Comm comm, char *name, PetscScalar *data, PetscInt n);

void writetofile_f2f(MPI_Comm comm, char *name, double *data, PetscInt n);

PetscErrorCode saveVecHDF5(MPI_Comm comm, Vec vec, const char *filename, const char *dsetname);

PetscErrorCode saveVecMfile(MPI_Comm comm, Vec vec, const char *filename, const char *dsetname);

#endif
