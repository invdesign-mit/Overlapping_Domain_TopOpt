#include "sym.h"

extern int count;

#undef __FUNCT__
#define __FUNCT__ "mirrorxy"
void mirrorxy(PetscReal *u, PetscReal *uext, int nx,int ny,int nlayers,int nxcells,int nycells, int mirrorXY[2], int transpose)
{

  int mirrorX = mirrorXY[0];
  int mirrorY = mirrorXY[1];
  
  for(int icy=0; icy<nycells; icy++){
    for(int icx=0; icx<nxcells; icx++){
      for(int iy=0; iy<ny; iy++){
	for(int ix=0; ix<nx; ix++){
	  for(int il=0; il<nlayers; il++){

	    int i = il + nlayers*ix + nlayers*nx*iy + nlayers*nx*ny*icx + nlayers*nx*ny*nxcells*icy;

	    int iix = ix + nx*icx;
	    int iiy = iy + ny*icy;

	    int jjx1 = iix;
	    int jjx2 = 2*nx*nxcells - iix - 1;
	    int jjy1 = iiy;
	    int jjy2 = 2*ny*nycells - iiy - 1;

	    int jx1  = jjx1 % nx;
	    int jcx1 = jjx1 / nx;
	    int jx2  = jjx2 % nx;
	    int jcx2 = jjx2 / nx;
	    int jy1  = jjy1 % ny;
	    int jcy1 = jjy1 / ny;
	    int jy2  = jjy2 % ny;
	    int jcy2 = jjy2 / ny;

	    int jl = il;

	    int nnxcells = (mirrorX==1) ? 2*nxcells : nxcells;
	    
	    int j11 = jl + nlayers*jx1 + nlayers*nx*jy1 + nlayers*nx*ny*jcx1 + nlayers*nx*ny*nnxcells*jcy1;
	    int j21 = jl + nlayers*jx2 + nlayers*nx*jy1 + nlayers*nx*ny*jcx2 + nlayers*nx*ny*nnxcells*jcy1;
	    int j12 = jl + nlayers*jx1 + nlayers*nx*jy2 + nlayers*nx*ny*jcx1 + nlayers*nx*ny*nnxcells*jcy2;
	    int j22 = jl + nlayers*jx2 + nlayers*nx*jy2 + nlayers*nx*ny*jcx2 + nlayers*nx*ny*nnxcells*jcy2;

	    if(transpose==0){

	      uext[j11] = u[i];
	      if(mirrorX==1)
		uext[j21] = u[i];
	      if(mirrorY==1)
		uext[j12] = u[i];
	      if(mirrorX==1 && mirrorY==1)
		uext[j22] = u[i];
		
	    }else{

	      u[i] = uext[j11];
	      if(mirrorX==1)
		u[i] += uext[j21];
	      if(mirrorY==1)
		u[i] += uext[j12];
	      if(mirrorX==1 && mirrorY==1)
		u[i] += uext[j22];
		  
	    }

	    
	  }
	}
      }
    }
  }

	    
}


#undef __FUNCT__
#define __FUNCT__ "ffintensitysym"
double ffintensitysym(int ndof, double *dof, double *grad, void *data)
{

  data_ *ptdata = (data_ *) data;

  int mirrorX = ptdata->mirrorXY[0];
  int mirrorY = ptdata->mirrorXY[1];
  int mirrorXY[2] = {mirrorX,mirrorY};

  DOFInfo dofi = ptdata->dofi;
  int nx = dofi.nx;
  int ny = dofi.ny;
  int nlayers = dofi.numlayers;
  int nxcells = (mirrorX==1) ? dofi.numcells_x/2 : dofi.numcells_x;
  int nycells = (mirrorY==1) ? dofi.numcells_y/2 : dofi.numcells_y;
  int neps_total = dofi.neps_total;

  double *dofext=(double *)malloc(neps_total*sizeof(double));
  double *gradext=(double *)malloc(neps_total*sizeof(double));
  mirrorxy(dof, dofext, nx,ny,nlayers,nxcells,nycells, mirrorXY, 0);
  MPI_Barrier(PETSC_COMM_WORLD);
  
  double objval = ffintensity(neps_total,dofext,gradext,data);
  mirrorxy(grad, gradext, nx,ny,nlayers,nxcells,nycells, mirrorXY, 1); 

  free(dofext);
  free(gradext);

  return objval;
}

#undef __FUNCT__
#define __FUNCT__ "ffintensitysym_maximinconstraint"
double ffintensitysym_maximinconstraint(int ndof_with_dummy, double *dof_with_dummy, double *grad_with_dummy, void *data)
{
  int ndof=ndof_with_dummy-1;
  double obj=ffintensitysym(ndof,&(dof_with_dummy[0]),&(grad_with_dummy[0]),data);

  for(int i=0;i<ndof;i++){
    grad_with_dummy[i]=-1.0*grad_with_dummy[i];
  }
  grad_with_dummy[ndof]=1.0;

  count--;

  return dof_with_dummy[ndof]-obj;

}
