#ifndef GUARD_obj_h
#define GUARD_obj_h

#include "petsc.h"
#include "type.h"
#include "grid.h"

typedef struct{

  Mat W;

  PetscReal filter_beta;
  PetscReal filter_eta;
  
  int iz_src;
  int iz_mtr;
  
  PetscReal freq;
  PetscReal omega;
  PetscScalar omega_complex;

  Mat CurlCurl;
  Mat Curl;

  KSP *ksp;
  int maxit;
  int *its;

  Vec *x;
  
  PetscScalar **Jx;
  PetscScalar **Jy;

  PetscScalar **ux;
  PetscScalar **uy;
  PetscScalar **vx;
  PetscScalar **vy;
  PetscScalar **wx;
  PetscScalar **wy;

  Vec epsDiff;
  Vec epsBkg;

  MPI_Comm subcomm;
  int colour;
  int specID;
  DOFInfo dofi;
  GridInfo gi;
  ParDataGrid dg;

  int print_at_singleobj;
  int print_at_multiobj;

  int mirrorXY[2];

  PetscScalar total_phaseoverlap[3];
  
} data_;

#endif
