#include "phprofile.h"

#undef __FUNCT__
#define __FUNCT__ "phdiff"
void phdiff(MPI_Comm big_comm, int colour, int ncells_per_comm, PetscScalar **vx, PetscScalar **vy, int nx, int ny, int px, int py, int numcells_x, int numcells_y, double dx, double dy, double freq, double oxy[2], double xyzfar[3], double axy[2], PetscBool conjugate)
{

  int numlayers=1;

  int mx=nx+2*px;
  int my=ny+2*py;

  int Ntot = nx*ny*numlayers*numcells_x*numcells_y;
  int Mtot = mx*my*numlayers*numcells_x*numcells_y;
    
  Mat W;
  ovmat(big_comm, &W, nx,ny, px,py, numcells_x,numcells_y, numlayers, 0, 1,1);
  Vec vn,vm;
  MatCreateVecs(W,&vn,&vm);

  PetscScalar *unx = (PetscScalar *) malloc(Ntot*sizeof(PetscScalar));
  PetscScalar *uny = (PetscScalar *) malloc(Ntot*sizeof(PetscScalar));
  PetscScalar *umx = (PetscScalar *) malloc(Mtot*sizeof(PetscScalar));
  PetscScalar *umy = (PetscScalar *) malloc(Mtot*sizeof(PetscScalar));

  for(int jcy=0;jcy<numcells_y;jcy++){
    for(int jcx=0;jcx<numcells_x;jcx++){
      for(int jy=0;jy<ny;jy++){
	for(int jx=0;jx<nx;jx++){
	  for(int jl=0;jl<numlayers;jl++){

	    int j = jl + numlayers*jx + numlayers*nx*jy + numlayers*nx*ny*jcx + numlayers*nx*ny*numcells_x*jcy;

	    double x = ((double)(jx+nx*jcx) + 0.5)*dx - oxy[0];
	    double y = ((double)(jy+ny*jcy) + 0.5)*dy - oxy[1];
	    double z = 0;
	    double xfar = xyzfar[0];
	    double yfar = xyzfar[1];
	    double zfar = xyzfar[2];
	    
	    double r2f = sqrt( pow(x-xfar,2) + pow(y-yfar,2) + pow(z-zfar,2) );
	    
	    PetscScalar expfac = cexp( - PETSC_i * 2 * M_PI * freq * (r2f - zfar) );
	    
	    unx[j] = axy[0]*expfac;
	    uny[j] = axy[1]*expfac;

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
