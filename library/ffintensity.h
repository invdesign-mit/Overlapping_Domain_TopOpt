#ifndef GUARD_ffintensity_h
#define GUARD_ffintensity_h

#include "petsc.h"
#include "type.h"
#include "ovmat.h"
#include "input.h"
#include "output.h"
#include "planewave.h"
#include "grid.h"
#include "obj.h"
#include "dof2dom.h"
#include "array2vec.h"
#include "logging.h"
#include "pml.h"
#include "vec.h"
#include "mat.h"
#include "solver.h"
#include "filters.h"

double ffintensity(int ndof, double *dof, double *grad, void *data);

double ffintensity_maximinconstraint(int ndof_with_dummy, double *dof_with_dummy, double *grad_with_dummy, void *data);

#endif
