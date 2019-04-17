#ifndef GUARD_filters_h
#define GUARD_filters_h

#include "petsc.h"
#include "type.h"

void density_filter(MPI_Comm comm, Mat *Qout, int nx, int ny, int numcells_x, int numcells_y, int numlayers, double rx, double ry, double alpha, int normalized);

void threshold_projection_filter(Vec rho_in, Vec rho_out, Vec rho_grad, double filter_threshold, double filter_steepness);

#endif
