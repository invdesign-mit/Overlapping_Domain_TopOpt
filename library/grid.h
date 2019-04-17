#ifndef GUARD_grid_h
#define GUARD_grid_h

#include "petsc.h"
#include "hdf5.h"
#include "type.h"
#include "input.h"

typedef struct {

  PetscInt N[Naxis];  // # of grid points in x, y, z
  PetscInt Ntot;  // total # of unknowns
  BC bc[Naxis]; // boundary conditions at -x, -y, -z ends; for simplicity, we set BC+ == BC-
  PetscScalar *dl[Naxis][Ngt];  // dx, dy, dz at primary and dual grid locations
  PetscInt Npml[Naxis][Nsign]; //number of pml layers along x, y, z at + and - ends.
  PetscScalar exp_neg_ikL[Naxis];  // exp(-ik Lx), exp(-ik Ly), exp(-ik Lz)

} GridInfo;

typedef struct{

  DM da;  // distributed array
  PetscInt Nlocal_tot;  // total # of local unknowns
  PetscInt Nlocal[Naxis];  // # of local grid points in x, y, z
  PetscInt start[Naxis]; // local starting points in x, y, z
  PetscInt Nlocal_g[Naxis];  // # of local grid points in x, y, z including ghost points
  PetscInt start_g[Naxis]; // local starting points in x, y, z including ghost points
  Vec vecTemp; // template vector.  Also used as a temporary storage of a vector
  ISLocalToGlobalMapping map;  // local-to-global index mapping

} ParDataGrid;

typedef struct{

  int nx;
  int ny;
  int px;
  int py;
  int numcells_x;
  int numcells_y;
  int numlayers;
  int mx;
  int my;
  int *mz;
  int mxo;
  int myo;
  int *mzo;
  int mzslab;
  int ncells_per_comm;
  int ncells_total;
  int neps_per_layer;
  int neps_per_cell;
  int neps_per_comm;
  int meps_per_layer;
  int meps_per_cell;
  int meps_per_comm;
  int ncomms;
  int neps_total;
  int meps_total;

} DOFInfo;

typedef struct{

  KrylovType solverID; // 0 BiCG, 1 QMR 
  PetscBool use_mat_sym; //check if matrix is symmetric and, if true, use a symalg
  PetscInt max_iter;  // maximum number of iteration of BiCG
  PetscReal tol;  // tolerance of BiCG
  PetscInt relres_interval;  // number of BiCG iterations between snapshots of approximate solutions

} IterSolverInfo;

void setDOFInfo(MPI_Comm comm, int id, DOFInfo *dofi,
		int nx, int ny, int numlayers, int numcells_x, int numcells_y,
		int *mz, int *mzo, int mzslab,
		int ncells_per_comm, int ncomms);


PetscErrorCode setGridInfo(MPI_Comm comm, const char *inputfile_name, GridInfo *gi);

PetscErrorCode setParDataGrid(MPI_Comm comm, ParDataGrid *dg, GridInfo gi);

PetscErrorCode setIterSolverInfo(const char *flag_prefix, IterSolverInfo *si);

#endif
