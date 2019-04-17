#include "array2vec.h"

#undef __FUNCT__
#define __FUNCT__ "array2mpi_c2c"
PetscErrorCode array2mpi_c2c(PetscScalar *pt, Vec v)
{
  PetscErrorCode ierr;
  int j, ns, ne;

  ierr = VecGetOwnershipRange(v,&ns,&ne);
  for(j=ns;j<ne;j++)
    { ierr=VecSetValue(v,j,pt[j],INSERT_VALUES);
      CHKERRQ(ierr);
    }

  ierr = VecAssemblyBegin(v); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(v);  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "mpi2array_c2c"
PetscErrorCode mpi2array_c2c(Vec v, PetscScalar *pt, int n)
{
  PetscErrorCode ierr;
  int i;
  PetscScalar *_a;
  Vec V_SEQ;
  VecScatter ctx;

  ierr = VecScatterCreateToAll(v,&ctx,&V_SEQ);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,v,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,v,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(V_SEQ,&_a);CHKERRQ(ierr);
  for (i = 0; i < n; i++) pt[i] = _a[i];
  ierr = VecRestoreArray(V_SEQ,&_a);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&V_SEQ);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "array2mpi_f2c"
PetscErrorCode array2mpi_f2c(PetscReal *pt, Vec v)
{
  PetscErrorCode ierr;
  int j, ns, ne;

  ierr = VecGetOwnershipRange(v,&ns,&ne);
  for(j=ns;j<ne;j++)
    { ierr=VecSetValue(v,j,pt[j]+PETSC_i*0.0,INSERT_VALUES);
      CHKERRQ(ierr);
    }

  ierr = VecAssemblyBegin(v); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(v);  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "mpi2array_c2f"
PetscErrorCode mpi2array_c2f(Vec v, PetscReal *pt, int n)
{
  PetscErrorCode ierr;
  int i;
  PetscScalar *_a;
  Vec V_SEQ;
  VecScatter ctx;

  ierr = VecScatterCreateToAll(v,&ctx,&V_SEQ);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,v,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,v,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(V_SEQ,&_a);CHKERRQ(ierr);
  for (i = 0; i < n; i++) pt[i] = creal(_a[i]);
  ierr = VecRestoreArray(V_SEQ,&_a);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&V_SEQ);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

