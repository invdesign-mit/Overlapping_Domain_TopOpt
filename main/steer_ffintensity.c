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
    for(int i=0;i<dofi[ispec].ncells_per_comm;i++){
      data[ispec].ux[i]=(PetscScalar *)malloc(gi[ispec].N[Xx]*gi[ispec].N[Yy]*sizeof(PetscScalar));
      data[ispec].uy[i]=(PetscScalar *)malloc(gi[ispec].N[Xx]*gi[ispec].N[Yy]*sizeof(PetscScalar));
    }
    sprintf(tmpstr,"-spec%d_out_kx,ky,ax,ay",ispec);
    getrealarray(tmpstr,kaxy,&nget,1);
    kx=kaxy[0], ky=kaxy[1];
    ax=kaxy[2]*cexp(PETSC_i*kaxy[3]);
    ay=kaxy[4]*cexp(PETSC_i*kaxy[5]);
    conjugate=PETSC_TRUE;
    planewave(PETSC_COMM_WORLD, colour, ncells_per_comm,
	      data[ispec].ux, data[ispec].uy,
	      dofi[ispec].nx, dofi[ispec].ny,
	      dofi[ispec].px, dofi[ispec].py,
	      dofi[ispec].numcells_x, dofi[ispec].numcells_y,
	      0.0,dx,dy,
	      kx,ky, ax,ay,
	      conjugate);
    MPI_Barrier(PETSC_COMM_WORLD);

    data[ispec].vx=NULL, data[ispec].vy=NULL;
    data[ispec].wx=NULL, data[ispec].wy=NULL;
    
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
    
  }

  PetscReal *dof=(PetscReal *)malloc(neps_total*sizeof(PetscReal));
  char init_filename[PETSC_MAX_PATH_LEN];
  getstr("-init_dof_name",init_filename,"dof.txt");
  readfromfile_f2f(init_filename,dof,neps_total);

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

    PetscReal *grad=(PetscReal *)malloc(neps_total*sizeof(PetscReal));    
    
    double ss[4];
    int ns=4;
    getrealarray("-is,s0,s1,ds",ss,&ns,0);
    int is=round(ss[0]);
    double s0=ss[1], s1=ss[2], ds=ss[3];
    for(double s=s0;s<s1;s+=ds){
      //for(int i=0;i<15;i++){
      dof[is]=s;
      double objval = ffintensity(neps_total,dof,grad,&(data[specID]));
      PetscPrintf(PETSC_COMM_WORLD,"objval: %g %.16g %.16g \n",dof[is],objval,grad[is]);
    }

    free(grad);
    
  }

  if(Job==1){

    int specID[nspecs];
    int numopt=nspecs;
    getintarray("-specID", specID, &numopt, 0);

    if(numopt==1){
    
      double *lb=(double *)malloc(neps_total*sizeof(double));
      double *ub=(double *)malloc(neps_total*sizeof(double));
      for(int i=0;i<neps_total;i++){
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

      double result=optimize_generic(neps_total, dof,
				     lb, ub,
				     (nlopt_func)ffintensity, &(data[specID[0]]),
				     NULL,NULL,0,
				     alg,
				     &nlopt_return);

      PetscPrintf(PETSC_COMM_WORLD,"nlopt return value: %d \n",nlopt_return);
      PetscPrintf(PETSC_COMM_WORLD,"optimal objval: %0.8g \n",result);

      free(lb);
      free(ub);
      
    }else if(numopt>1){

      int ndofAll=neps_total+1;
      double *dofAll=(double *)malloc(ndofAll*sizeof(double));
      double *lbAll=(double *)malloc(ndofAll*sizeof(double));
      double *ubAll=(double *)malloc(ndofAll*sizeof(double));
      for(int i=0;i<neps_total;i++){
	lbAll[i]=0.0;
	ubAll[i]=1.0;
	dofAll[i]=dof[i];
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
	maximins[i]=(nlopt_func)ffintensity_maximinconstraint;
	data[specID[i]].print_at_singleobj=-1;
	constrdata[i]=&(data[specID[i]]);
      }

      int *print_at=&print_at_multiobj;
      
      double result=optimize_generic(ndofAll, dofAll,
				     lbAll, ubAll,
				     (nlopt_func)dummy_obj, print_at,
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
  
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscFinalize();
  return 0;
}
