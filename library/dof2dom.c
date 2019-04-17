#include "dof2dom.h"

#undef __FUNCT__
#define __FUNCT__ "multilayer_forward"
PetscErrorCode multilayer_forward(PetscScalar *pt, Vec v, DOFInfo *dofi, DM da)
{

  PetscInt nlayers = dofi->numlayers;
  PetscInt Mx = dofi->mx;
  PetscInt My = dofi->my;
  PetscInt *Mz = dofi->mz;
  PetscInt Mzslab = dofi->mzslab;
  PetscInt Mxo = dofi->mxo;
  PetscInt Myo = dofi->myo;
  PetscInt *Mzo = dofi->mzo;
  PetscScalar val;

  PetscErrorCode ierr;

  Field ***v_array;  // 3D array that images Vec v
  ierr = DMDAVecGetArray(da, v, &v_array); CHKERRQ(ierr);

  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscInt ix,iy,iz,ic,px,py,pz,ilayer,indp;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
	for (ic = 0; ic < Naxis; ++ic) {
	  /** v_array is just the array-representation of the vector v.  So,
	      setting values on v_array is actually setting values on v.*/
	  val=0.0;
	  if(ix>=Mxo && ix<Mxo+Mx){
	    px=ix-Mxo;
	    if(iy>=Myo && iy<Myo+My){
	      py=iy-Myo;
	      for(ilayer=0;ilayer<nlayers;ilayer++){
		if(iz>=Mzo[ilayer] && iz<Mzo[ilayer]+Mz[ilayer]){
		    pz = (Mzslab==0) ? iz-Mzo[ilayer] : 0 ;
		    indp = ilayer + nlayers*px + nlayers*Mx*py + nlayers*Mx*My*pz;
		    val=pt[indp];
		}
	      }
	    }
	  }
	  v_array[iz][iy][ix].comp[ic]=val;
		    
	}
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v, &v_array); CHKERRQ(ierr);



  return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "multilayer_backward"
PetscErrorCode multilayer_backward(MPI_Comm comm, Vec v, PetscScalar *pt, DOFInfo *dofi, DM da)
{

  PetscInt nlayers = dofi->numlayers;
  PetscInt Mx = dofi->mx;
  PetscInt My = dofi->my;
  PetscInt *Mz = dofi->mz;
  PetscInt Mzslab = dofi->mzslab;
  PetscInt Mxo = dofi->mxo;
  PetscInt Myo = dofi->myo;
  PetscInt *Mzo = dofi->mzo;
  PetscInt ndof = dofi->meps_per_cell;

  PetscErrorCode ierr;

  PetscInt i;
  PetscScalar *local_pt;
  local_pt = (PetscScalar *) malloc(ndof*sizeof(PetscScalar));
  for(i=0;i<ndof;i++) local_pt[i]=0.0+PETSC_i*0.0;
  MPI_Barrier(comm);

  Field ***v_array;  // 3D array that images Vec v
  ierr = DMDAVecGetArray(da, v, &v_array); CHKERRQ(ierr);

  /** Get corners and widths of Yee's grid included in this proces. */
  PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
  PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
  ierr = DMDAGetCorners(da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

  PetscInt ix,iy,iz,ic,px,py,pz,ilayer,indp;
  for (iz = oz; iz < oz+nz; ++iz) {
    for (iy = oy; iy < oy+ny; ++iy) {
      for (ix = ox; ix < ox+nx; ++ix) {
	for (ic = 0; ic < Naxis; ++ic) {
	  if(ix>=Mxo && ix<Mxo+Mx){
	    px=ix-Mxo;
	    if(iy>=Myo && iy<Myo+My){
	      py=iy-Myo;
	      for(ilayer=0;ilayer<nlayers;ilayer++){
		if(iz>=Mzo[ilayer] && iz<Mzo[ilayer]+Mz[ilayer]){
		    pz = (Mzslab==0) ? iz-Mzo[ilayer] : 0 ;
		    indp = ilayer + nlayers*px + nlayers*Mx*py + nlayers*Mx*My*pz;
		    local_pt[indp] = local_pt[indp]+v_array[iz][iy][ix].comp[ic];
		}
	      }
	    }
	  }
		    
	}
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, v, &v_array); CHKERRQ(ierr);
  MPI_Barrier(comm);

  MPI_Allreduce(local_pt,pt,ndof,MPIU_SCALAR,MPI_SUM,comm);
  MPI_Barrier(comm);

  free(local_pt);

  return ierr;
}
