#include "planewave.h"

#undef __FUNCT__
#define __FUNCT__ "planewave"
void planewave(MPI_Comm big_comm, int colour, int ncells_per_comm, PetscScalar **vx, PetscScalar **vy, int nx, int ny, int px, int py, int numcells_x, int numcells_y, PetscScalar val_margin, double dx, double dy, double kx, double ky, PetscScalar ax, PetscScalar ay, PetscBool conjugate)
{

  int numlayers=1;

  int mx=nx+2*px;
  int my=ny+2*py;

  int Ntot = nx*ny*numlayers*numcells_x*numcells_y;
  int Mtot = mx*my*numlayers*numcells_x*numcells_y;

  double kLx=kx*(nx*numcells_x)*dx;
  double kLy=ky*(ny*numcells_y)*dy;
  double x,y;
    
  Mat W;
  ovmat(big_comm, &W, nx,ny, px,py, numcells_x,numcells_y, numlayers, val_margin, kLx,kLy);
  Vec vn,vm;
  MatCreateVecs(W,&vn,&vm);

  PetscScalar *unx = (PetscScalar *) malloc(Ntot*sizeof(PetscScalar));
  PetscScalar *uny = (PetscScalar *) malloc(Ntot*sizeof(PetscScalar));
  PetscScalar *umx = (PetscScalar *) malloc(Mtot*sizeof(PetscScalar));
  PetscScalar *umy = (PetscScalar *) malloc(Mtot*sizeof(PetscScalar));

  int jl,jx,jy,jcx,jcy,j;
  for(jcy=0;jcy<numcells_y;jcy++){
    for(jcx=0;jcx<numcells_x;jcx++){
      for(jy=0;jy<ny;jy++){
	for(jx=0;jx<nx;jx++){
	  for(jl=0;jl<numlayers;jl++){

	    j = jl + numlayers*jx + numlayers*nx*jy + numlayers*nx*ny*jcx + numlayers*nx*ny*numcells_x*jcy;

	    x = ((double)(jx+nx*jcx) + 0.5)*dx;
	    y = ((double)(jy+ny*jcy) + 0.5)*dy;

	    unx[j] = ax*cexp( -PETSC_i * (kx*x + ky*y) );
	    uny[j] = ay*cexp( -PETSC_i * (kx*x + ky*y) );

	  }
	}
      }
    }
  }
  MPI_Barrier(big_comm);
  
  array2mpi_c2c(unx, vn);
  MatMult(W,vn,vm);
  mpi2array_c2c(vm,umx,Mtot);

  array2mpi_c2c(uny, vn);
  MatMult(W,vn,vm);
  mpi2array_c2c(vm,umy,Mtot);
  
  for(int i=0;i<ncells_per_comm;i++){
    for(int j=0;j<mx*my;j++){
      if(conjugate){
	vx[i][j]=conj(umx[j+mx*my*(i+ncells_per_comm*colour)]);
	vy[i][j]=conj(umy[j+mx*my*(i+ncells_per_comm*colour)]);
      }else{
	vx[i][j]=umx[j+mx*my*(i+ncells_per_comm*colour)];
	vy[i][j]=umy[j+mx*my*(i+ncells_per_comm*colour)];
      }
    }
  }
  
  MatDestroy(&W);
  VecDestroy(&vn);
  VecDestroy(&vm);

  free(unx);
  free(uny);
  free(umx);
  free(umy);
  
  MPI_Barrier(big_comm);

}
