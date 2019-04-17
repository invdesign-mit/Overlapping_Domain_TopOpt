#include "ovmat.h"

#undef __FUNCT__
#define __FUNCT__ "ovmat"
void ovmat(MPI_Comm comm, Mat *Wout, int nx, int ny, int px, int py, int numcells_x, int numcells_y, int numlayers, PetscScalar val_margin, double kLx, double kLy)
{

  PetscPrintf(comm,"Creating the overlap extension matrix.\n");

  int mx=2*px+nx;
  int my=2*py+ny;
  int Mx=mx*numcells_x;
  int My=my*numcells_y;
  int mrows=Mx*My*numlayers;

  int Nx=nx*numcells_x;
  int Ny=ny*numcells_y;
  int ncols=Nx*Ny*numlayers;
  
  Mat W;

  int ns,ne;
  int i,j,k;
  int ix,iy,il,icx,icy;
  int jx,jy,jl,jcx,jcy;
  PetscScalar val;
  
  MatCreate(comm,&W);
  MatSetType(W,MATMPIAIJ);
  MatSetSizes(W,PETSC_DECIDE,PETSC_DECIDE, mrows,ncols);
  MatMPIAIJSetPreallocation(W, 1, PETSC_NULL, 1, PETSC_NULL);

  MatGetOwnershipRange(W, &ns, &ne);

  for(i=ns;i<ne;i++){

    val=1.0;
    
    il=(k=i)%numlayers;
    ix=(k/=numlayers)%mx;
    iy=(k/=mx)%my;
    icx=(k/=my)%numcells_x;
    icy=(k/=numcells_x)%numcells_y;

    if(ix < px){
      jcx=icx-1;
      jx = ix - (px-1) + (nx-1);
      val *= val_margin;
    }else if(ix >= px+nx){
      jcx=icx+1;
      jx = ix - (px+nx);
      val *= val_margin;
    }else{
      jcx=icx;
      jx = ix - px;
    }
    if(jcx<0){
      jcx+=numcells_x;
      val*=cexp( PETSC_i*kLx);
    }
    if(jcx>=numcells_x){
      jcx-=numcells_x;
      val*=cexp(-PETSC_i*kLx);
    }
    
    if(iy < py){
      jcy=icy-1;
      jy = iy - (py-1) + (ny-1);
      val *= val_margin;
    }else if(iy >= py+ny){
      jcy=icy+1;
      jy = iy - (py+ny);
      val *= val_margin;
    }else{
      jcy=icy;
      jy = iy - py;
    }
    if(jcy<0){
      jcy+=numcells_y;
      val*=cexp( PETSC_i*kLy);
    }
    if(jcy>=numcells_y){
      jcy-=numcells_y;
      val*=cexp(-PETSC_i*kLy);
    }

    jl=il;
    j=jl + numlayers*jx + numlayers*nx*jy + numlayers*nx*ny*jcx + numlayers*nx*ny*numcells_x*jcy;
      
    MatSetValue(W,i,j,val,INSERT_VALUES);

  }

  MatAssemblyBegin(W, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(W, MAT_FINAL_ASSEMBLY);

  *Wout = W;

}

#undef __FUNCT__
#define __FUNCT__ "vecfill_zslice"
PetscErrorCode vecfill_zslice(MPI_Comm comm, DM da, int mx, int my, PetscScalar *vx, PetscScalar *vy, PetscScalar *vz, Vec v, int iz0)
{

  PetscErrorCode ierr;

  VecSet(v,0.0);

  Field ***v_array;  // 3D array that images Vec v
  ierr = DMDAVecGetArray(da, v, &v_array); CHKERRQ(ierr);

  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt lx, ly, lz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &lx, &ly, &lz); CHKERRQ(ierr);

  PetscInt ix,iy,iz,i;
  for (iz = oz; iz < oz+lz; ++iz) {
    for (iy = oy; iy < oy+ly; ++iy) {
      for (ix = ox; ix < ox+lx; ++ix) {
	if(iz==iz0){
	  i = ix + mx*iy;
	  if(vx) v_array[iz][iy][ix].comp[0]=vx[i];
	  if(vy) v_array[iz][iy][ix].comp[1]=vy[i];
	  if(vz) v_array[iz][iy][ix].comp[2]=vz[i];
	}

      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v, &v_array); CHKERRQ(ierr);

  return ierr;

}
