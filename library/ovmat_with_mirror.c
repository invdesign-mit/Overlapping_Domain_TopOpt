#include "ovmat_with_mirror.h"

#undef __FUNCT__
#define __FUNCT__ "ovmatsym"
void ovmatsym(MPI_Comm comm, Mat *Wout, int nx, int ny, int px, int py, int numcells_x, int numcells_y, int numlayers, PetscScalar val_margin, double kLx, double kLy, int mirrorXY[2])
{

  int mirrorX = mirrorXY[0];
  int mirrorY = mirrorXY[1];
  PetscPrintf(comm,"Creating the overlap extension matrix. Mirror boundary option at the (-x,-y) end is available: Given %d,%d.\n",mirrorX,mirrorY);

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
      if(mirrorX==1){
	jcx=0;
	jx=px-ix-1;
	val=1;
      }else{
	jcx+=numcells_x;
	val*=cexp( PETSC_i*kLx);
      }
    }
    if(jcx>=numcells_x){
      if(mirrorX==1){
	jcx=numcells_x-1;
	jx=(nx-1)-ix+(px+nx);
	val=1;
      }else{
	jcx-=numcells_x;
	val*=cexp(-PETSC_i*kLx);
      }
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
      if(mirrorY==1){
	jcy=0;
	jy=py-iy-1;
	val=1;
      }else{
	jcy+=numcells_y;
	val*=cexp( PETSC_i*kLy);
      }
    }
    if(jcy>=numcells_y){
      if(mirrorY==1){
	jcy=numcells_y-1;
	jy=(ny-1)-iy+(py+ny);
	val=1;
      }else{
	jcy-=numcells_y;
	val*=cexp(-PETSC_i*kLy);
      }
    }

    jl=il;
    j=jl + numlayers*jx + numlayers*nx*jy + numlayers*nx*ny*jcx + numlayers*nx*ny*numcells_x*jcy;
      
    MatSetValue(W,i,j,val,INSERT_VALUES);

  }

  MatAssemblyBegin(W, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(W, MAT_FINAL_ASSEMBLY);

  *Wout = W;

}
