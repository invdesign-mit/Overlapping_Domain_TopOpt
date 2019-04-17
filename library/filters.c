#include "filters.h"

#undef __FUNCT__
#define __FUNCT__ "density_filter"
void density_filter(MPI_Comm comm, Mat *Qout, int nx, int ny, int numcells_x, int numcells_y, int numlayers, double rx, double ry, double alpha, int normalized)
{

  PetscPrintf(comm,"Creating the density filter. NOTE: rx and ry must be greater than 0. rx,ry <= 1 means no filter.\n");

  int Nx=nx*numcells_x;
  int Ny=ny*numcells_y;
  int ncols=Nx*Ny*numlayers;
  int nrows=ncols;
  
  Mat Q;

  int pre_box_size=(2*ceil(rx)-1)*(2*ceil(ry)-1);
  
  MatCreate(comm,&Q);
  MatSetType(Q,MATMPIAIJ);
  MatSetSizes(Q,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  MatMPIAIJSetPreallocation(Q, pre_box_size, PETSC_NULL, pre_box_size, PETSC_NULL);

  int ns,ne;
  MatGetOwnershipRange(Q, &ns, &ne);

  int box_size=(2*ceil(rx)-1)*(2*ceil(ry)-1);
  int *cols = (int *)malloc(box_size*sizeof(int));
  PetscScalar *weights = (PetscScalar *)malloc(box_size*sizeof(PetscScalar));

  for(int i=ns;i<ne;i++){

    int k;
    int il=(k=i)%numlayers;
    int ix=(k/=numlayers)%nx;
    int iy=(k/=nx)%ny;
    int icx=(k/=ny)%numcells_x;
    int icy=(k/=numcells_x)%numcells_y;

    int iix=ix+nx*icx;
    int iiy=iy+ny*icy;

    int jjx_min=iix-ceil(rx)+1;
    int jjx_max=iix+ceil(rx);
    int jjy_min=iiy-ceil(ry)+1;
    int jjy_max=iiy+ceil(ry);

    int ind=0;
    PetscScalar norm=0.0+PETSC_i*0.0;
    
    for(int jjy=jjy_min;jjy<jjy_max;jjy++){
      for(int jjx=jjx_min;jjx<jjx_max;jjx++){
	
	int jjjx, jjjy;

	if(jjx < 0)
	  jjjx = jjx + Nx;
	else if(jjx >= Nx)
	  jjjx = jjx - Nx;
	else
	  jjjx = jjx;

	if(jjy < 0)
	  jjjy = jjy + Ny;
	else if(jjy >= Ny)
	  jjjy = jjy - Ny;
	else
	  jjjy = jjy;

	int jx=jjjx%nx;
	int jcx=jjjx/nx;
	int jy=jjjy%ny;
	int jcy=jjjy/ny;

	int j=il + numlayers*jx + numlayers*nx*jy + numlayers*nx*ny*jcx + numlayers*nx*ny*numcells_x*jcy;
	PetscReal dist2=pow(iix-jjx,2)+pow(iiy-jjy,2);
	PetscReal alpha2=pow(alpha,2);
	cols[ind]=j;
	weights[ind]=exp(-dist2/alpha2)+PETSC_i*0.0;
	norm += weights[ind];
	ind++;
	
      }
    }
    
    if(normalized==1)
      for(int j=0;j<ind;j++) weights[j]/=norm;

    MatSetValues(Q, 1, &i, ind, cols, weights, INSERT_VALUES);
    
  }

  MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY);

  free(cols);
  free(weights);
  
  *Qout = Q;

}

#undef __FUNCT__
#define __FUNCT__ "threshold_projection_filter"
void threshold_projection_filter(Vec rho_in, Vec rho_out, Vec rho_grad, double filter_threshold, double filter_steepness)
{

  PetscReal eta=filter_threshold;
  PetscReal beta=filter_steepness;

  VecSet(rho_out,0.0);
  VecSet(rho_grad,0.0);

  PetscScalar *rin,*rout,*rg;
  VecGetArray(rho_in,&rin);
  VecGetArray(rho_out,&rout);
  VecGetArray(rho_grad,&rg);

  int i,ns,ne;
  VecGetOwnershipRange(rho_in, &ns, &ne);

  for(i=ns;i<ne;i++){

    if(beta<1e-3){

      rout[i-ns]=rin[i-ns];
      rg[i-ns]=1.0+PETSC_i*0.0;

    }else{
      
      PetscReal rho= creal(rin[i-ns]);
      PetscReal r1 = tanh(beta*eta) + tanh(beta*(rho-eta));
      PetscReal r2 = tanh(beta*eta) + tanh(beta*(1.0-eta));
      rout[i-ns]= r1/r2 + PETSC_i*0.0;
      PetscReal r3 = beta * cosh(beta*eta) * cosh(beta-beta*eta) / sinh(beta);
      rg[i-ns]  = r3 / ( cosh(beta*(rho-eta)) * cosh(beta*(rho-eta)) ) + PETSC_i*0.0;
    }

  }

  VecRestoreArray(rho_in,&rin);
  VecRestoreArray(rho_out,&rout);
  VecRestoreArray(rho_grad,&rg);

}
