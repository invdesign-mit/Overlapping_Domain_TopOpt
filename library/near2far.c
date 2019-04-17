#include "near2far.h"

void green3d(PetscScalar *EH, const double *x,
	     double freq, double eps, double mu,
	     const double *x0, int comp, PetscScalar f0)
{

  double rhat[3] = {x[0]-x0[0],x[1]-x0[1],x[2]-x0[2]};
  double r = sqrt(rhat[0]*rhat[0]+rhat[1]*rhat[1]+rhat[2]*rhat[2]);
  rhat[0]=rhat[0]/r, rhat[1]=rhat[1]/r, rhat[2]=rhat[2]/r;

  double n = sqrt(eps*mu);
  double k = 2*M_PI*freq*n;
  PetscScalar ikr = PETSC_i * k*r;
  double ikr2   = -(k*r)*(k*r);
  /* note that SCUFF-EM computes the fields from the dipole moment p,
       whereas we need it from the current J = -i*omega*p, so our result
       is divided by -i*omega compared to SCUFF */
  PetscScalar expfac = f0 * (k*n/(4*M_PI*r)) * cexp( PETSC_i*(k*r + M_PI*0.5) );
  double Z = sqrt(mu/eps);

  double p[3]={0,0,0};
  p[comp%3]=1.0;
  double pdotrhat = p[0]*rhat[0] + p[1]*rhat[1] + p[2]*rhat[2];
  
  double rhatcrossp[3] = {rhat[1] * p[2] -
			  rhat[2] * p[1],
			  rhat[2] * p[0] -
			  rhat[0] * p[2],
			  rhat[0] * p[1] -
			  rhat[1] * p[0]};
  
  /* compute the various scalar quantities in the point source formulae */
  PetscScalar term1 =  1.0 - 1.0/ikr + 1.0/ikr2;
  PetscScalar term2 = (-1.0 + 3.0/ikr - 3.0/ikr2) * pdotrhat;
  PetscScalar term3 = (1.0 - 1.0/ikr);
  /* now assemble everything based on source type */
  if (comp<3) {
    expfac /= eps;
    EH[0] = expfac * (term1*p[0] + term2*rhat[0]);
    EH[1] = expfac * (term1*p[1] + term2*rhat[1]);
    EH[2] = expfac * (term1*p[2] + term2*rhat[2]);
    EH[3] = expfac*term3*rhatcrossp[0] / Z;
    EH[4] = expfac*term3*rhatcrossp[1] / Z;
    EH[5] = expfac*term3*rhatcrossp[2] / Z;
  }
  else {
    expfac /= mu;
    EH[0] = -expfac*term3*rhatcrossp[0] * Z;
    EH[1] = -expfac*term3*rhatcrossp[1] * Z;
    EH[2] = -expfac*term3*rhatcrossp[2] * Z;
    EH[3] = expfac * (term1*p[0] + term2*rhat[0]);
    EH[4] = expfac * (term1*p[1] + term2*rhat[1]);
    EH[5] = expfac * (term1*p[2] + term2*rhat[2]);
  }

  //conjugation
  EH[0]=conj(EH[0]);
  EH[1]=conj(EH[1]);
  EH[2]=conj(EH[2]);
  EH[3]=conj(EH[3]);
  EH[4]=conj(EH[4]);
  EH[5]=conj(EH[5]);
  
}


void create_near2far(MPI_Comm big_comm, int colour, int ncells_per_comm,
		     PetscScalar **ux, PetscScalar **uy,
		     PetscScalar **vx, PetscScalar **vy,
		     PetscScalar **wx, PetscScalar **wy,
		     int nx, int ny, int px, int py, int numcells_x, int numcells_y, double dx, double dy,
		     double oxy[2], int symxy[2],
		     double xyzfar[3],
		     double freq, double eps, double mu, PetscBool conjugate)
{

  PetscScalar Fex[6],Fey[6];
  int Nxy=nx*ny*numcells_x*numcells_y;
  PetscScalar *vx_cx=(PetscScalar *)malloc(Nxy*sizeof(PetscScalar));
  PetscScalar *vy_cx=(PetscScalar *)malloc(Nxy*sizeof(PetscScalar));
  PetscScalar *vz_cx=(PetscScalar *)malloc(Nxy*sizeof(PetscScalar));
  PetscScalar *vx_cy=(PetscScalar *)malloc(Nxy*sizeof(PetscScalar));
  PetscScalar *vy_cy=(PetscScalar *)malloc(Nxy*sizeof(PetscScalar));
  PetscScalar *vz_cy=(PetscScalar *)malloc(Nxy*sizeof(PetscScalar));  

  for(int icy=0;icy<numcells_y;icy++){
    for(int icx=0;icx<numcells_x;icx++){
      for(int iy=0;iy<ny;iy++){
	for(int ix=0;ix<nx;ix++){
	  
	  PetscScalar Mx =  1.0; //get the right magnetic currents
	  PetscScalar My = -1.0; //get the right magnetic currents

	  int i = ix + nx*iy + nx*ny*icx + nx*ny*icy;
	  
	  double x = ((double)(ix+nx*icx) + 0.5)*dx;
	  double y = ((double)(iy+ny*icy) + 0.5)*dy;
	  
	  double xyznear[3]={ x - oxy[0], y - oxy[1], 0.0 };
	  green3d(Fey, xyzfar, freq,eps,mu, xyznear, 3, Mx);
	  green3d(Fex, xyzfar, freq,eps,mu, xyznear, 4, My);
	  vx_cx[i]=Fex[0], vx_cy[i]=Fey[0];
	  vy_cx[i]=Fex[1], vy_cy[i]=Fey[1];
	  vz_cx[i]=Fex[2], vz_cy[i]=Fey[2];

	  if(symxy[0]==1){ //mirror fields at (-x,y,z); even boundary 
	    double xyz[3]={ -xyznear[0], xyznear[1], xyznear[2] };
	    green3d(Fey, xyzfar, freq,eps,mu, xyz, 3, Mx);
	    green3d(Fex, xyzfar, freq,eps,mu, xyz, 4,-My);
	    vx_cx[i]+=Fex[0], vx_cy[i]+=Fey[0];
	    vy_cx[i]+=Fex[1], vy_cy[i]+=Fey[1];
	    vz_cx[i]+=Fex[2], vz_cy[i]+=Fey[2];
	  }
	  if(symxy[0]==-1){ //mirror fields at (-x,y,z); odd boundary 
	    double xyz[3]={ -xyznear[0], xyznear[1], xyznear[2] };
	    green3d(Fey, xyzfar, freq,eps,mu, xyz, 3,-Mx);
	    green3d(Fex, xyzfar, freq,eps,mu, xyz, 4, My);
	    vx_cx[i]+=Fex[0], vx_cy[i]+=Fey[0];
	    vy_cx[i]+=Fex[1], vy_cy[i]+=Fey[1];
	    vz_cx[i]+=Fex[2], vz_cy[i]+=Fey[2];
	  }
	  if(symxy[1]==1){ //mirror fields at (x,-y,z); even boundary 
	    double xyz[3]={  xyznear[0],-xyznear[1], xyznear[2] };
	    green3d(Fey, xyzfar, freq,eps,mu, xyz, 3,-Mx);
	    green3d(Fex, xyzfar, freq,eps,mu, xyz, 4, My);
	    vx_cx[i]+=Fex[0], vx_cy[i]+=Fey[0];
	    vy_cx[i]+=Fex[1], vy_cy[i]+=Fey[1];
	    vz_cx[i]+=Fex[2], vz_cy[i]+=Fey[2];
	  }
	  if(symxy[1]==-1){ //mirror fields at (x,-y,z); odd boundary 
	    double xyz[3]={  xyznear[0],-xyznear[1], xyznear[2] };
	    green3d(Fey, xyzfar, freq,eps,mu, xyz, 3, Mx);
	    green3d(Fex, xyzfar, freq,eps,mu, xyz, 4,-My);
	    vx_cx[i]+=Fex[0], vx_cy[i]+=Fey[0];
	    vy_cx[i]+=Fex[1], vy_cy[i]+=Fey[1];
	    vz_cx[i]+=Fex[2], vz_cy[i]+=Fey[2];
	  }
	  if( (symxy[0]==1 && symxy[1]==1) || (symxy[0]==-1 && symxy[1]==-1) ){ //mirror fields at (-x,-y,z); only even/odd boundaries 
	    double xyz[3]={ -xyznear[0],-xyznear[1], xyznear[2] };
	    green3d(Fey, xyzfar, freq,eps,mu, xyz, 3,-Mx);
	    green3d(Fex, xyzfar, freq,eps,mu, xyz, 4,-My);
	    vx_cx[i]+=Fex[0], vx_cy[i]+=Fey[0];
	    vy_cx[i]+=Fex[1], vy_cy[i]+=Fey[1];
	    vz_cx[i]+=Fex[2], vz_cy[i]+=Fey[2];
	  }
	  if( (symxy[0]==1 && symxy[1]==-1) || (symxy[0]==-1 && symxy[1]==1) ){ //mirror fields at (-x,-y,z); mixed boundaries 
	    double xyz[3]={ -xyznear[0],-xyznear[1], xyznear[2] };
	    green3d(Fey, xyzfar, freq,eps,mu, xyz, 3, Mx);
	    green3d(Fex, xyzfar, freq,eps,mu, xyz, 4, My);
	    vx_cx[i]+=Fex[0], vx_cy[i]+=Fey[0];
	    vy_cx[i]+=Fex[1], vy_cy[i]+=Fey[1];
	    vz_cx[i]+=Fex[2], vz_cy[i]+=Fey[2];
	  }


	}
      }
    }
  }
  
  int numlayers=1;

  int mx=nx+2*px;
  int my=ny+2*py;
  int Mtot = mx*my*numlayers*numcells_x*numcells_y;

  Mat W;
  ovmat(big_comm, &W, nx,ny, px,py, numcells_x,numcells_y, numlayers, 0, 1,1);
  Vec tmp_n,tmp_m;
  MatCreateVecs(W,&tmp_n,&tmp_m);

  PetscScalar *umx = (PetscScalar *) malloc(Mtot*sizeof(PetscScalar));
  PetscScalar *umy = (PetscScalar *) malloc(Mtot*sizeof(PetscScalar));
  PetscScalar *vmx = (PetscScalar *) malloc(Mtot*sizeof(PetscScalar));
  PetscScalar *vmy = (PetscScalar *) malloc(Mtot*sizeof(PetscScalar));
  PetscScalar *wmx = (PetscScalar *) malloc(Mtot*sizeof(PetscScalar));
  PetscScalar *wmy = (PetscScalar *) malloc(Mtot*sizeof(PetscScalar));

  MPI_Barrier(big_comm);

  array2mpi_c2c(vx_cx, tmp_n);
  MatMult(W,tmp_n,tmp_m);
  mpi2array_c2c(tmp_m,umx,Mtot);
  array2mpi_c2c(vx_cy, tmp_n);
  MatMult(W,tmp_n,tmp_m);
  mpi2array_c2c(tmp_m,umy,Mtot);

  array2mpi_c2c(vy_cx, tmp_n);
  MatMult(W,tmp_n,tmp_m);
  mpi2array_c2c(tmp_m,vmx,Mtot);
  array2mpi_c2c(vy_cy, tmp_n);
  MatMult(W,tmp_n,tmp_m);
  mpi2array_c2c(tmp_m,vmy,Mtot);

  array2mpi_c2c(vz_cx, tmp_n);
  MatMult(W,tmp_n,tmp_m);
  mpi2array_c2c(tmp_m,wmx,Mtot);
  array2mpi_c2c(vz_cy, tmp_n);
  MatMult(W,tmp_n,tmp_m);
  mpi2array_c2c(tmp_m,wmy,Mtot);
  
  
  for(int i=0;i<ncells_per_comm;i++){
    for(int j=0;j<mx*my;j++){
      if(conjugate){
	ux[i][j]=conj(umx[j+mx*my*(i+ncells_per_comm*colour)]);
	uy[i][j]=conj(umy[j+mx*my*(i+ncells_per_comm*colour)]);
	vx[i][j]=conj(vmx[j+mx*my*(i+ncells_per_comm*colour)]);
	vy[i][j]=conj(vmy[j+mx*my*(i+ncells_per_comm*colour)]);
	wx[i][j]=conj(wmx[j+mx*my*(i+ncells_per_comm*colour)]);
	wy[i][j]=conj(wmy[j+mx*my*(i+ncells_per_comm*colour)]);
      }else{
	ux[i][j]=umx[j+mx*my*(i+ncells_per_comm*colour)];
	uy[i][j]=umy[j+mx*my*(i+ncells_per_comm*colour)];
	vx[i][j]=vmx[j+mx*my*(i+ncells_per_comm*colour)];
	vy[i][j]=vmy[j+mx*my*(i+ncells_per_comm*colour)];
	wx[i][j]=wmx[j+mx*my*(i+ncells_per_comm*colour)];
	wy[i][j]=wmy[j+mx*my*(i+ncells_per_comm*colour)];
      }
    }
  }

  MatDestroy(&W);
  VecDestroy(&tmp_n);
  VecDestroy(&tmp_m);

  free(vx_cx);
  free(vx_cy);
  free(vy_cx);
  free(vy_cy);
  free(vz_cx);
  free(vz_cy);

  free(umx);
  free(umy);
  free(vmx);
  free(vmy);
  free(wmx);
  free(wmy);
  
  MPI_Barrier(big_comm);
  
  
}


