#include "petsc.h"
#include "petscsys.h"
#include "hdf5.h"
#include "nlopt.h"
#include <assert.h>
#include "libFDOPT.h"

int count=0;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{

  MPI_Init(NULL, NULL);
  PetscInitialize(&argc,&argv,NULL,NULL);

  int numcells_x, numcells_y;
  int ncells_per_comm, ncells_total;
  int ncomms;
  int nproc_per_comm;
  int size,nproc_total;
  
  getint("-numcells_x",&numcells_x,1);
  getint("-numcells_y",&numcells_y,1);
  getint("-ncells_per_comm",&ncells_per_comm,1);
  getint("-nproc_per_comm",&nproc_per_comm,1);

  ncells_total=numcells_x*numcells_y;
  ncomms=ncells_total/ncells_per_comm;

  MPI_Comm_size(MPI_COMM_WORLD,&size);
  nproc_total = ncomms * nproc_per_comm;
  PetscPrintf(PETSC_COMM_WORLD, "\tNOTE: nproc_total = (ncells_x*ncells_y/ncells_per_comm) * nproc_per_comm = %d\n",nproc_total);
  if(!(nproc_total == size)) SETERRQ(PETSC_COMM_WORLD,1,"The total number of processors is not consistent with the given configuration.");
  PetscPrintf(PETSC_COMM_WORLD,
	      "\tThe total # of procs is %d.\n\tThe total # of cells is %d: %d cells along x and %d cells along y.\n\tThe # of subcomms is %d, each with %d core(s).\n\tEach subcomm sequentially handles %d cells.\n",
	      nproc_total,
	      ncells_total,
	      numcells_x,
	      numcells_y,
	      ncomms,
	      nproc_per_comm,
	      ncells_per_comm);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm subcomm;
  int colour = rank/nproc_per_comm;
  MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &subcomm);
  
  int nx,ny,numlayers;
  getint("-nx",&nx,60);
  getint("-ny",&ny,40);
  getint("-numlayers",&numlayers,3);
  int *mz=(int *)malloc(numlayers*sizeof(mz));
  int *mzo=(int *)malloc(numlayers*sizeof(mzo));
  getintarray("-mzo",mzo,&numlayers,100);
  getintarray("-mz",mz,&numlayers,10);
  PetscReal dx,dy,dz;
  getreal("-dx",&dx,0.02);
  getreal("-dy",&dy,0.02);
  getreal("-dz",&dz,0.02);
  int mzslab;
  getint("-mzslab",&mzslab,1);

  int mirrorX, mirrorY;
  getint("-mirrorX",&mirrorX,1);
  getint("-mirrorY",&mirrorY,0);
  
  int iz_src, iz_mtr;
  getint("-iz_source",&iz_src,30);
  getint("-iz_monitor",&iz_mtr,130);

  int maxit;
  getint("-maxit",&maxit,15);

  int print_at_singleobj, print_at_multiobj;
  getint("-print_at_singleobj",&print_at_singleobj,-1);
  getint("-print_at_multiobj",&print_at_multiobj,-1);
  
  PetscReal filter_rx,filter_ry,filter_alpha;
  int filter_normalized;
  PetscReal filter_beta, filter_eta;
  getreal("-filter_rx",&filter_rx,1);
  getreal("-filter_ry",&filter_ry,1);
  getreal("-filter_alpha",&filter_alpha,100000);
  getint("-filter_normalized",&filter_normalized,1);
  getreal("-filter_beta",&filter_beta,0);
  getreal("-filter_eta",&filter_eta,0.5);
  
  Mat Q;
  density_filter(PETSC_COMM_WORLD, &Q, nx,ny,numcells_x,numcells_y,numlayers, filter_rx, filter_ry, filter_alpha, filter_normalized);
  
  int nspecs;
  getint("-nspecs",&nspecs,1);

  int neps_per_layer = nx * ny;
  int neps_per_cell = neps_per_layer * numlayers;
  int neps_per_comm = neps_per_cell * ncells_per_comm;
  int neps_total = neps_per_comm * ncomms;
  
  DOFInfo *dofi=(DOFInfo *)malloc(nspecs*sizeof(DOFInfo));
  GridInfo *gi=(GridInfo *)malloc(nspecs*sizeof(GridInfo));
  ParDataGrid *dg=(ParDataGrid *)malloc(nspecs*sizeof(ParDataGrid));
  data_ *data=(data_ *)malloc(nspecs*sizeof(data_));
  for(int ispec=0;ispec<nspecs;ispec++){

    setDOFInfo(PETSC_COMM_WORLD, ispec, dofi+ispec,
	       nx,ny,numlayers,numcells_x,numcells_y,
	       mz,mzo,mzslab,
	       ncells_per_comm,ncomms);

    Mat Wtmp;
    ovmat(PETSC_COMM_WORLD, &Wtmp,
	  nx, ny,
	  dofi[ispec].px, dofi[ispec].py,
	  numcells_x, numcells_y, numlayers,
	  1, 0, 0);
    MatMatMult(Wtmp,Q,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&(data[ispec].W));
    MatDestroy(&Wtmp);
    
    char tmpstr[PETSC_MAX_PATH_LEN];
    sprintf(tmpstr,"grid%d.h5",ispec);
    setGridInfo(PETSC_COMM_WORLD, tmpstr, gi+ispec);
    setParDataGrid(subcomm, dg+ispec, gi[ispec]);
    MPI_Barrier(PETSC_COMM_WORLD);

    data[ispec].iz_src=iz_src;
    data[ispec].iz_mtr=iz_mtr;
    
    data[ispec].Jx=(PetscScalar **)malloc(dofi[ispec].ncells_per_comm*sizeof(PetscScalar *));
    data[ispec].Jy=(PetscScalar **)malloc(dofi[ispec].ncells_per_comm*sizeof(PetscScalar *));
    for(int i=0;i<dofi[ispec].ncells_per_comm;i++){
      data[ispec].Jx[i]=(PetscScalar *)malloc(gi[ispec].N[Xx]*gi[ispec].N[Yy]*sizeof(PetscScalar));
      data[ispec].Jy[i]=(PetscScalar *)malloc(gi[ispec].N[Xx]*gi[ispec].N[Yy]*sizeof(PetscScalar));
    }

    sprintf(tmpstr,"-spec%d_freq",ispec);
    getreal(tmpstr,&(data[ispec].freq),1.0);
    data[ispec].omega=2.0*M_PI*data[ispec].freq;
    data[ispec].omega_complex=data[ispec].omega + PETSC_i * 0.0;
    
    PetscReal kaxy[6];
    int nget=6;
    sprintf(tmpstr,"-spec%d_kx,ky,ax,ay",ispec);
    getrealarray(tmpstr,kaxy,&nget,1);
    PetscReal kx=kaxy[0], ky=kaxy[1];
    PetscScalar ax=kaxy[2]*cexp(PETSC_i*kaxy[3]);
    PetscScalar ay=kaxy[4]*cexp(PETSC_i*kaxy[5]);
    PetscBool conjugate=PETSC_FALSE;
    planewave(PETSC_COMM_WORLD, colour, ncells_per_comm,
	      data[ispec].Jx, data[ispec].Jy,
	      dofi[ispec].nx, dofi[ispec].ny,
	      dofi[ispec].px, dofi[ispec].py,
	      dofi[ispec].numcells_x, dofi[ispec].numcells_y,
	      1.0,dx,dy,
	      kx,ky, ax,ay,
	      conjugate);
    MPI_Barrier(PETSC_COMM_WORLD);

    data[ispec].ux=(PetscScalar **)malloc(dofi[ispec].ncells_per_comm*sizeof(PetscScalar *));
    data[ispec].uy=(PetscScalar **)malloc(dofi[ispec].ncells_per_comm*sizeof(PetscScalar *));
    data[ispec].vx=(PetscScalar **)malloc(dofi[ispec].ncells_per_comm*sizeof(PetscScalar *));
    data[ispec].vy=(PetscScalar **)malloc(dofi[ispec].ncells_per_comm*sizeof(PetscScalar *));
    data[ispec].wx=(PetscScalar **)malloc(dofi[ispec].ncells_per_comm*sizeof(PetscScalar *));
    data[ispec].wy=(PetscScalar **)malloc(dofi[ispec].ncells_per_comm*sizeof(PetscScalar *));
    for(int i=0;i<dofi[ispec].ncells_per_comm;i++){
      data[ispec].ux[i]=(PetscScalar *)malloc(gi[ispec].N[Xx]*gi[ispec].N[Yy]*sizeof(PetscScalar));
      data[ispec].uy[i]=(PetscScalar *)malloc(gi[ispec].N[Xx]*gi[ispec].N[Yy]*sizeof(PetscScalar));
      data[ispec].vx[i]=(PetscScalar *)malloc(gi[ispec].N[Xx]*gi[ispec].N[Yy]*sizeof(PetscScalar));
      data[ispec].vy[i]=(PetscScalar *)malloc(gi[ispec].N[Xx]*gi[ispec].N[Yy]*sizeof(PetscScalar));
      data[ispec].wx[i]=(PetscScalar *)malloc(gi[ispec].N[Xx]*gi[ispec].N[Yy]*sizeof(PetscScalar));
      data[ispec].wy[i]=(PetscScalar *)malloc(gi[ispec].N[Xx]*gi[ispec].N[Yy]*sizeof(PetscScalar));
    }
    PetscReal far_params[7];
    nget=7;
    sprintf(tmpstr,"-spec%d_out_oxy,symxy,xyzfar",ispec);
    getrealarray(tmpstr,far_params,&nget,0);
    double oxy[2]={far_params[0],far_params[1]};
    int symxy[2]={round(far_params[2]),round(far_params[3])};
    double xyzfar[3]={far_params[4],far_params[5],far_params[6]};
    conjugate=PETSC_FALSE;
    create_near2far(PETSC_COMM_WORLD, colour, ncells_per_comm,
		    data[ispec].ux, data[ispec].uy,
		    data[ispec].vx, data[ispec].vy,
		    data[ispec].wx, data[ispec].wy,
		    nx, ny, dofi[ispec].px, dofi[ispec].py, numcells_x, numcells_y, dx, dy,
		    oxy, symxy,
		    xyzfar,
		    data[ispec].freq, 1.0,1.0, conjugate);
    MPI_Barrier(PETSC_COMM_WORLD);
       
    sprintf(tmpstr,"epsDiff%d.h5",ispec);
    VecDuplicate(dg[ispec].vecTemp,&(data[ispec].epsDiff));
    VecSet(data[ispec].epsDiff,0.0);
    loadVecHDF5(subcomm,data[ispec].epsDiff,tmpstr,"/eps");
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);

    sprintf(tmpstr,"epsBkg%d.h5",ispec);
    VecDuplicate(dg[ispec].vecTemp,&(data[ispec].epsBkg));
    VecSet(data[ispec].epsBkg,0.0);
    loadVecHDF5(subcomm,data[ispec].epsBkg,tmpstr,"/eps");
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);

    Vec mu;
    VecDuplicate(dg[ispec].vecTemp,&mu);
    VecSet(mu,1.0);
    create_doublecurl_op(subcomm, &(data[ispec].CurlCurl), &(data[ispec].Curl), data[ispec].omega_complex, mu, gi[ispec], dg[ispec]);
    VecDestroy(&mu);
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);

    data[ispec].ksp=(KSP *)malloc(ncells_per_comm*sizeof(KSP));
    data[ispec].maxit=maxit;
    data[ispec].its=(int *)malloc(ncells_per_comm*sizeof(KSP));
    for(int i=0;i<ncells_per_comm;i++){
      setupKSPDirect(subcomm,&(data[ispec].ksp[i]),data[ispec].maxit);
      data[ispec].its[i]=100;
    }

    data[ispec].x=(Vec *)malloc(ncells_per_comm*sizeof(Vec));
    for(int i=0;i<ncells_per_comm;i++)
      VecDuplicate(dg[ispec].vecTemp,&(data[ispec].x[i]));

    data[ispec].subcomm=subcomm;
    data[ispec].colour=colour;
    data[ispec].specID=ispec;
    data[ispec].dofi=dofi[ispec];
    data[ispec].gi=gi[ispec];
    data[ispec].dg=dg[ispec];

    data[ispec].filter_beta=filter_beta;
    data[ispec].filter_eta=filter_eta;

    data[ispec].print_at_singleobj=print_at_singleobj;
    data[ispec].print_at_multiobj=print_at_multiobj;

    data[ispec].mirrorXY[0]=mirrorX;
    data[ispec].mirrorXY[1]=mirrorY;
    
  }

  //read arbitrary dof input
  PetscReal *dof=(PetscReal *)malloc(neps_total*sizeof(PetscReal));
  char init_filename[PETSC_MAX_PATH_LEN];
  getstr("-init_dof_name",init_filename,"dof.txt");
  readfromfile_f2f(init_filename,dof,neps_total);

  //pick up a quadrant of dof, discarding the rest; this quadrant (not the full dof) should be used as the initial guess for the optimization
  int nxcells = (mirrorX==1) ? numcells_x/2 : numcells_x;
  int nycells = (mirrorY==1) ? numcells_y/2 : numcells_y;
  int reduceXY[2] = {0,0};
  int neps_reduced = nx*ny*numlayers*nxcells*nycells;
  PetscReal *dof_reduced=(PetscReal *)malloc(neps_reduced*sizeof(PetscReal));
  mirrorxy(dof_reduced,dof, nx,ny,numlayers, nxcells,nycells, reduceXY,1);

  //re-symmetrize the dof array
  int mirrorXY[2] = {mirrorX,mirrorY};
  mirrorxy(dof_reduced,dof, nx,ny,numlayers, nxcells,nycells, mirrorXY,0);
  
  int Job;
  getint("-Job",&Job,0);

  if(Job==0){

    int specID;
    getint("-specID",&specID,0);
    int printEfield;
    getint("-printEfield",&printEfield,0);

    PetscScalar *dofext=(PetscScalar *)malloc(dofi[specID].meps_total*sizeof(PetscScalar));
    Vec _dofext, _dof;
    MatCreateVecs(data[specID].W,&_dof,&_dofext);
    array2mpi_f2c(dof,_dof);
    MatMult(data[specID].W,_dof,_dofext);
    Vec rho_out,rho_grad;
    VecDuplicate(_dofext,&rho_out);
    VecDuplicate(_dofext,&rho_grad);
    threshold_projection_filter(_dofext,rho_out,rho_grad,filter_eta,filter_beta);
    mpi2array_c2c(rho_out,dofext,dofi[specID].meps_total);
    
    for(int i=0;i<ncells_per_comm;i++){
      int j=i+ncells_per_comm*colour;
      int jx=j%numcells_x;
      int jy=j/numcells_x;

      PetscPrintf(subcomm,"Printing epsilon at cell ID %d %d \n",jx,jy);
    
      Vec eps;
      VecDuplicate(dg[specID].vecTemp,&eps);
      multilayer_forward(&(dofext[j*dofi[specID].meps_per_cell]), eps, &dofi[specID], dg[specID].da);
      VecPointwiseMult(eps,eps,data[specID].epsDiff);
      VecAXPY(eps,1.0,data[specID].epsBkg);

      char tmpstr[PETSC_MAX_PATH_LEN];
      sprintf(tmpstr,"epscell%d_%d.h5",jx,jy);
      saveVecHDF5(subcomm,eps,tmpstr,"eps");

      if(printEfield){

	PetscPrintf(subcomm,"Computing and printing Efield at cell ID %d %d \n",jx,jy);
	
	Mat M; 
	MatDuplicate(data[specID].CurlCurl,MAT_COPY_VALUES,&M);
	VecScale(eps,-pow(data[specID].omega,2));
	MatDiagonalSet(M,eps,ADD_VALUES);

	Vec b;
	VecDuplicate(data[specID].dg.vecTemp,&b);
	vecfill_zslice(data[specID].subcomm, data[specID].dg.da, data[specID].dofi.mx,data[specID].dofi.my, data[specID].Jx[i],data[specID].Jy[i], NULL, b, data[specID].iz_src);
	VecScale(b,-PETSC_i*data[specID].omega);

	SolveMatrixDirect(data[specID].subcomm,data[specID].ksp[i],M,b,data[specID].x[i],&(data[specID].its[i]),data[specID].maxit);
	MPI_Barrier(subcomm);
	MPI_Barrier(PETSC_COMM_WORLD);

	sprintf(tmpstr,"Efield%d_%d.h5",jx,jy);
	saveVecHDF5(subcomm,data[specID].x[i],tmpstr,"E");

	VecDestroy(&b);
	MatDestroy(&M);
	
      }

      VecDestroy(&eps);
      
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);
      
    }
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);
    free(dofext);
    VecDestroy(&rho_out);
    VecDestroy(&rho_grad);
    VecDestroy(&_dofext);
    VecDestroy(&_dof);
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);

  }

  if(Job==-1){

    int specID;
    getint("-specID",&specID,0);

    PetscReal *grad=(PetscReal *)malloc(neps_reduced*sizeof(PetscReal));    
    
    double ss[4];
    int ns=4;
    getrealarray("-is,s0,s1,ds",ss,&ns,0);
    int is=round(ss[0]);
    double s0=ss[1], s1=ss[2], ds=ss[3];
    for(double s=s0;s<s1;s+=ds){
      dof[is]=s;
      double objval = ffintensitysym(neps_reduced,dof_reduced,grad,&(data[specID]));
      PetscPrintf(PETSC_COMM_WORLD,"objval: %g %.16g %.16g \n",dof[is],objval,grad[is]);
    }

    free(grad);
    
  }

  if(Job==1){

    int specID[nspecs];
    int numopt=nspecs;
    getintarray("-specID", specID, &numopt, 0);

    if(numopt==1){
    
      double *lb=(double *)malloc(neps_reduced*sizeof(double));
      double *ub=(double *)malloc(neps_reduced*sizeof(double));
      for(int i=0;i<neps_reduced;i++){
	lb[i]=0.0;
	ub[i]=1.0;
      }
      MPI_Barrier(PETSC_COMM_WORLD);

      int algouter, alginner, algmaxeval;
      getint("-algouter",&algouter,24);
      getint("-alginner",&alginner,24);
      getint("-algmaxeval",&algmaxeval,500);
      alg_ alg={(nlopt_algorithm)algouter,(nlopt_algorithm)alginner,algmaxeval,1000000,1};

      nlopt_result nlopt_return;

      double result=optimize_generic(neps_reduced, dof_reduced,
				     lb, ub,
				     (nlopt_func)ffintensitysym, &(data[specID[0]]),
				     NULL,NULL,0,
				     alg,
				     &nlopt_return);

      PetscPrintf(PETSC_COMM_WORLD,"nlopt return value: %d \n",nlopt_return);
      PetscPrintf(PETSC_COMM_WORLD,"optimal objval: %0.8g \n",result);

      free(lb);
      free(ub);
      
    }else if(numopt>1){

      int ndofAll=neps_reduced+1;
      double *dofAll=(double *)malloc(ndofAll*sizeof(double));
      double *lbAll=(double *)malloc(ndofAll*sizeof(double));
      double *ubAll=(double *)malloc(ndofAll*sizeof(double));
      for(int i=0;i<neps_reduced;i++){
	lbAll[i]=0.0;
	ubAll[i]=1.0;
	dofAll[i]=dof_reduced[i];
      }
      lbAll[ndofAll-1]=0.0;
      ubAll[ndofAll-1]=1.0/0.0;
      getreal("-initial_dummy",&(dofAll[ndofAll-1]),0.0);
      MPI_Barrier(PETSC_COMM_WORLD);

      int algouter, alginner, algmaxeval;
      getint("-algouter",&algouter,24);
      getint("-alginner",&alginner,24);
      getint("-algmaxeval",&algmaxeval,500);
      alg_ alg={(nlopt_algorithm)algouter,(nlopt_algorithm)alginner,algmaxeval,1000000,1};

      nlopt_result nlopt_return;

      void *constrdata[numopt];
      nlopt_func* maximins=(nlopt_func*)malloc(numopt*sizeof(nlopt_func));
      for(int i=0;i<numopt;i++){
	maximins[i]=(nlopt_func)ffintensitysym_maximinconstraint;
	data[specID[i]].print_at_singleobj=-1;
	constrdata[i]=&(data[specID[i]]);
      }

      double result=optimize_generic(ndofAll, dofAll,
				     lbAll, ubAll,
				     (nlopt_func)dummy_objsym, &(data[specID[0]]),
				     maximins,constrdata,numopt,
				     alg,
				     &nlopt_return);

      PetscPrintf(PETSC_COMM_WORLD,"nlopt return value: %d \n",nlopt_return);
      PetscPrintf(PETSC_COMM_WORLD,"optimal objval: %0.8g \n",result);

      free(dofAll);
      free(lbAll);
      free(ubAll);

    }
      
  }

  if(Job==2){

    int specID;
    getint("-specID",&specID,0);
    double far_cen[3],far_size[3],far_dh[3];
    int ffget=3;
    getrealarray("-ffwindow_cen",far_cen,&ffget,0);
    getrealarray("-ffwindow_size",far_size,&ffget,1);
    getrealarray("-ffwindow_dh",far_dh,&ffget,0.04);
    double tmp[4];
    ffget=4;
    getrealarray("-ffwindow_oxy,symxy",tmp,&ffget,0);
    double ffxmin=far_cen[0]-far_size[0]/2.0, ffdx=far_dh[0];
    double ffymin=far_cen[1]-far_size[1]/2.0, ffdy=far_dh[1];
    double ffzmin=far_cen[2]-far_size[2]/2.0, ffdz=far_dh[2];
    int ffnx=round(far_size[0]/far_dh[0]);
    int ffny=round(far_size[1]/far_dh[1]);
    int ffnz=round(far_size[2]/far_dh[2]);
    double ffoxy[2]={tmp[0],tmp[1]};
    int ffsymxy[2]={round(tmp[2]),round(tmp[3])};
    char ffname[PETSC_MAX_PATH_LEN];
    getstr("-ffwindow_outputfilename",ffname,"farfield_intensity.dat");
    int ideal_farfield;
    getint("-ideal_farfield",&ideal_farfield,-1);
    
    PetscScalar *dofext=(PetscScalar *)malloc(dofi[specID].meps_total*sizeof(PetscScalar));
    
    Vec _dofext, _dof;
    MatCreateVecs(data[specID].W,&_dof,&_dofext);
    array2mpi_f2c(dof,_dof);
    MatMult(data[specID].W,_dof,_dofext);
    Vec rho_out,rho_grad;
    VecDuplicate(_dofext,&rho_out);
    VecDuplicate(_dofext,&rho_grad);
    threshold_projection_filter(_dofext,rho_out,rho_grad,filter_eta,filter_beta);
    mpi2array_c2c(rho_out,dofext,dofi[specID].meps_total);

    for(int i=0;i<ncells_per_comm;i++){
      int j=i+ncells_per_comm*colour;
      Vec eps;
      VecDuplicate(dg[specID].vecTemp,&eps);
      multilayer_forward(&(dofext[j*dofi[specID].meps_per_cell]), eps, &dofi[specID], dg[specID].da);
      VecPointwiseMult(eps,eps,data[specID].epsDiff);
      VecAXPY(eps,1.0,data[specID].epsBkg);

      Mat M;
      MatDuplicate(data[specID].CurlCurl,MAT_COPY_VALUES,&M);
      VecScale(eps,-pow(data[specID].omega,2));
      MatDiagonalSet(M,eps,ADD_VALUES);

      Vec b;
      VecDuplicate(data[specID].dg.vecTemp,&b);
      vecfill_zslice(data[specID].subcomm, data[specID].dg.da, data[specID].dofi.mx,data[specID].dofi.my, data[specID].Jx[i],data[specID].Jy[i], NULL, b, data[specID].iz_src);
      VecScale(b,-PETSC_i*data[specID].omega);

      if(ideal_farfield>=0){
	
	if(ideal_farfield==0) //Ex-polarized farfield 
	  vecfill_zslice(data[specID].subcomm, data[specID].dg.da, data[specID].dofi.mx,data[specID].dofi.my, data[specID].ux[i],data[specID].uy[i], NULL, data[specID].x[i], data[specID].iz_mtr);
	if(ideal_farfield==1) //Ey-polarized farfield
	  vecfill_zslice(data[specID].subcomm, data[specID].dg.da, data[specID].dofi.mx,data[specID].dofi.my, data[specID].vx[i],data[specID].vy[i], NULL, data[specID].x[i], data[specID].iz_mtr);
	if(ideal_farfield==2) //Ez-polarized farfield
	  vecfill_zslice(data[specID].subcomm, data[specID].dg.da, data[specID].dofi.mx,data[specID].dofi.my, data[specID].wx[i],data[specID].wy[i], NULL, data[specID].x[i], data[specID].iz_mtr);
	VecConjugate(data[specID].x[i]);
	
      }else{

	SolveMatrixDirect(data[specID].subcomm,data[specID].ksp[i],M,b,data[specID].x[i],&(data[specID].its[i]),data[specID].maxit);

      }
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);
      
      VecDestroy(&b);
      MatDestroy(&M);
      VecDestroy(&eps);
      MPI_Barrier(subcomm);
      MPI_Barrier(PETSC_COMM_WORLD);

    }
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);
    free(dofext);
    VecDestroy(&rho_out);
    VecDestroy(&rho_grad);
    VecDestroy(&_dofext);
    VecDestroy(&_dof);
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);

    double *ffdata = (double *)malloc(ffnx*ffny*ffnz*sizeof(double));
    for(int iz=0;iz<ffnz;iz++){
      for(int iy=0;iy<ffny;iy++){
	for(int ix=0;ix<ffnx;ix++){
	  double x=ffxmin+(ix+0.5)*ffdx;
	  double y=ffymin+(iy+0.5)*ffdy;
	  double z=ffzmin+(iz+0.5)*ffdz;
	  double xyzfar[3]={x,y,z};

	  create_near2far(PETSC_COMM_WORLD, colour, ncells_per_comm,
			  data[specID].ux, data[specID].uy,
			  data[specID].vx, data[specID].vy,
			  data[specID].wx, data[specID].wy,
			  nx, ny, dofi[specID].px, dofi[specID].py, numcells_x, numcells_y, dx, dy,
			  ffoxy, ffsymxy,
			  xyzfar,
			  data[specID].freq, 1.0,1.0, PETSC_FALSE);
	  MPI_Barrier(PETSC_COMM_WORLD);
	  PetscScalar uobj_per_comm=0,vobj_per_comm=0,wobj_per_comm=0;
	  for(int i=0;i<ncells_per_comm;i++){
	    Vec uvec,vvec,wvec;
	    VecDuplicate(data[specID].x[i],&uvec);
	    VecDuplicate(data[specID].x[i],&vvec);
	    VecDuplicate(data[specID].x[i],&wvec);
	    vecfill_zslice(subcomm, dg[specID].da, dofi[specID].mx,dofi[specID].my, data[specID].ux[i],data[specID].uy[i], NULL, uvec, data[specID].iz_mtr);
	    vecfill_zslice(subcomm, dg[specID].da, dofi[specID].mx,dofi[specID].my, data[specID].vx[i],data[specID].vy[i], NULL, vvec, data[specID].iz_mtr);
	    vecfill_zslice(subcomm, dg[specID].da, dofi[specID].mx,dofi[specID].my, data[specID].wx[i],data[specID].wy[i], NULL, wvec, data[specID].iz_mtr);
	    PetscScalar tmpobj;
	    VecTDot(uvec,data[specID].x[i],&tmpobj);
	    uobj_per_comm+=tmpobj;
	    VecTDot(vvec,data[specID].x[i],&tmpobj);
	    vobj_per_comm+=tmpobj;
	    VecTDot(wvec,data[specID].x[i],&tmpobj);
	    wobj_per_comm+=tmpobj;
	    VecDestroy(&uvec);
	    VecDestroy(&vvec);
	    VecDestroy(&wvec);
	  }
	  MPI_Barrier(subcomm);
	  MPI_Barrier(PETSC_COMM_WORLD);

	  PetscScalar uobj,vobj,wobj;
	  int rank;
	  MPI_Comm_rank(subcomm, &rank);
	  if(rank!=0){
	    uobj_per_comm=0.0;
	    vobj_per_comm=0.0;
	    wobj_per_comm=0.0;
	  }
	  MPI_Allreduce(&uobj_per_comm,&uobj,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
	  MPI_Allreduce(&vobj_per_comm,&vobj,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
	  MPI_Allreduce(&wobj_per_comm,&wobj,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);

	  int ffip = ix+ffnx*iy+ffnx*ffny*iz;
	  double ffval=pow(cabs(uobj),2) + pow(cabs(vobj),2) + pow(cabs(wobj),2);
	  ffdata[ffip]=ffval;

	  PetscPrintf(PETSC_COMM_WORLD,"Farfield calculations [%g,%g,%g] %g ... %g%% complete.\n", x,y,z,ffval, ((double)ffip)/((double)ffnx*ffny*ffnz)*100.0);
	  
	}
      }
    }
    PetscPrintf(PETSC_COMM_WORLD,"***NOTE: ffwindow [nx,ny,nz] = [%d,%d,%d]\n",ffnx,ffny,ffnz);

    writetofile_f2f(PETSC_COMM_WORLD,ffname,ffdata,ffnx*ffny*ffnz);
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);

    free(ffdata);
    
  }


  
  
    
  free(mz);
  free(mzo);
  for(int ispec=0;ispec<nspecs;ispec++){
    free(dofi[ispec].mz);
    free(dofi[ispec].mzo);
    MatDestroy(&(data[ispec].W));
    MatDestroy(&(data[ispec].CurlCurl));
    MatDestroy(&(data[ispec].Curl));
    VecDestroy(&(data[ispec].epsDiff));
    VecDestroy(&(data[ispec].epsBkg));
    for(int i=0;i<ncells_per_comm;i++){
      free(data[ispec].Jx[i]);
      free(data[ispec].Jy[i]);
      if(data[ispec].ux) free(data[ispec].ux[i]);
      if(data[ispec].uy) free(data[ispec].uy[i]);
      if(data[ispec].vx) free(data[ispec].vx[i]);
      if(data[ispec].vy) free(data[ispec].vy[i]);
      if(data[ispec].wx) free(data[ispec].wx[i]);
      if(data[ispec].wy) free(data[ispec].wy[i]);
      KSPDestroy(&(data[ispec].ksp[i]));
      VecDestroy(&(data[ispec].x[i]));
    }
  }
  free(dofi);
  MatDestroy(&Q);

  free(dof);
  free(dof_reduced);
  
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscFinalize();
  return 0;
}
