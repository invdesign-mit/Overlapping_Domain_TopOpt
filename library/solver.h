#ifndef GUARD_solver_h
#define GUARD_solver_h

#include "grid.h"
#include "output.h"
#include "logging.h"
#include "mat.h"
#include "petsc.h"

typedef struct{

  KrylovType solverID; // 0 BiCG, 1 QMR
  PetscBool use_mat_sym; //check if matrix is symmetric and, if true, use a symalg
  PetscInt max_iter;  // maximum number of iteration of BiCG
  PetscReal tol;  // tolerance of BiCG
  PetscInt relres_interval;  // number of BiCG iterations between snapshots of approximate solutions

} SolverInfo;

typedef PetscErrorCode (*IterativeSolver)(MPI_Comm comm, const Mat A, Vec x, const Vec b, const PetscInt max_iter, const PetscReal tol, const PetscInt relres_interval);

PetscErrorCode bicg(MPI_Comm comm, const Mat A, Vec x, const Vec b, const PetscInt max_iter, const PetscReal tol, const PetscInt relres_interval);

PetscErrorCode bicgSymmetric(MPI_Comm comm, const Mat A, Vec x, const Vec b, const PetscInt max_iter, const PetscReal tol, const PetscInt relres_interval);

PetscErrorCode qmr(MPI_Comm comm, const Mat A, Vec x, const Vec b, const PetscInt max_iter, const PetscReal tol, const PetscInt relres_interval);

PetscErrorCode solveEq(MPI_Comm comm, Mat A, Vec x, Vec b, Vec LPC, Vec RPC, SolverInfo *si);

PetscErrorCode setupKSPDirect(MPI_Comm comm, KSP *kspout, int maxit);

PetscErrorCode SolveMatrixDirect(MPI_Comm comm, KSP ksp, Mat M, Vec b, Vec x, int *its, int maxit);

#endif
