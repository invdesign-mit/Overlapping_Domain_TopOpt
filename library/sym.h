#ifndef GUARD_sym_h
#define GUARD_sym_h

#include "petsc.h"
#include "ffintensity.h"
#include "obj.h"

void mirrorxy(PetscReal *u, PetscReal *uext, int nx,int ny,int nlayers,int nxcells,int nycells, int mirrorXY[2], int transpose);

double ffintensitysym(int ndof, double *dof, double *grad, void *data);

double ffintensitysym_maximinconstraint(int ndof_with_dummy, double *dof_with_dummy, double *grad_with_dummy, void *data);

#endif
