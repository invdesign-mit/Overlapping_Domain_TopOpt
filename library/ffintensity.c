#include "ffintensity.h"

extern int count;

PetscScalar objcell(MPI_Comm subcomm, PetscScalar *dofcell,
		    Mat CurlCurl, Vec epsDiff, Vec epsBkg, DOFInfo *dofi, DM da, PetscReal omega, Vec b, Vec x, KSP ksp, int *its, int maxit,
		    PetscScalar *ffx, PetscScalar *ffy, int iz_mtr, PetscScalar *grad_per_comm){

  Vec eps;
  VecDuplicate(b,&eps);
  multilayer_forward(dofcell, eps, dofi, da);
  VecPointwiseMult(eps,eps,epsDiff);
  VecAXPY(eps,1.0,epsBkg);

  Mat M;
  MatDuplicate(CurlCurl,MAT_COPY_VALUES,&M);
  VecScale(eps,-pow(omega,2));
  MatDiagonalSet(M,eps,ADD_VALUES);

  SolveMatrixDirect(subcomm,ksp,M,b,x,its,maxit);

  Vec ffvec;
  VecDuplicate(b,&ffvec);
  vecfill_zslice(subcomm, da, dofi->mx,dofi->my, ffx,ffy, NULL, ffvec, iz_mtr);

  PetscScalar obj;
  VecTDot(ffvec,x,&obj);
  
  Vec u,grad;
  VecDuplicate(b,&u);
  VecDuplicate(b,&grad);
  KSPSolveTranspose(ksp,ffvec,u);
  VecPointwiseMult(grad,u,x);

  VecScale(grad,omega*omega);
  VecPointwiseMult(grad,grad,epsDiff);
  multilayer_backward(subcomm,grad,grad_per_comm,dofi,da);


  VecDestroy(&eps);
  VecDestroy(&ffvec);
  VecDestroy(&u);
  VecDestroy(&grad);
  MatDestroy(&M);
  
  return obj;
	     
}

void consolidate(MPI_Comm subcomm, int colour, int meps_total, int meps_per_comm, PetscScalar obj_local, PetscScalar *obj_global, PetscScalar *grad_per_comm, double *grad){

  PetscScalar obj_per_comm=obj_local;
  PetscScalar obj_total;
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);
  
  int rank;
  MPI_Comm_rank(subcomm, &rank);
  if(rank!=0){
    obj_per_comm=0.0;
  }
  MPI_Allreduce(&obj_per_comm,&obj_total,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  
  double *tmp_grad=(double*) malloc(meps_total*sizeof(double));
  for(int i=0;i<meps_total;i++){
    int i_local=i-colour*meps_per_comm;
    if(rank==0 && 0<=i_local && i_local<meps_per_comm)
      tmp_grad[i] = 2.0*creal(conj(obj_total)*grad_per_comm[i_local]);
    else
      tmp_grad[i] = 0.0;
  }
  MPI_Allreduce(tmp_grad,grad,meps_total,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);

  free(tmp_grad);

  *obj_global=obj_total;

}

#undef __FUNCT__
#define __FUNCT__ "ffintensity"
double ffintensity(int ndof, double *dof, double *grad, void *data)
{

  PetscPrintf(PETSC_COMM_WORLD,"Computing the objective |int v . E|^2\n");
  
  data_ *ptdata = (data_ *) data;
  
  MPI_Comm subcomm = ptdata->subcomm;
  int colour = ptdata->colour;
  int specID = ptdata->specID;
  DOFInfo *dofi = &(ptdata->dofi);
  ParDataGrid dg = ptdata->dg;
  Mat W = ptdata->W;
  PetscReal filter_beta = ptdata->filter_beta; 
  PetscReal filter_eta = ptdata->filter_eta;
  Vec epsDiff = ptdata->epsDiff;
  Vec epsBkg = ptdata->epsBkg;
  Mat CurlCurl = ptdata->CurlCurl;
  PetscReal omega = ptdata->omega;
  PetscScalar **Jx = ptdata->Jx;
  PetscScalar **Jy = ptdata->Jy;
  PetscScalar **ux = ptdata->ux;
  PetscScalar **uy = ptdata->uy;
  PetscScalar **vx = ptdata->vx;
  PetscScalar **vy = ptdata->vy;
  PetscScalar **wx = ptdata->wx;
  PetscScalar **wy = ptdata->wy;
  int iz_src = ptdata->iz_src;
  int iz_mtr = ptdata->iz_mtr;
  Vec *x = ptdata->x;
  KSP *ksp = ptdata->ksp;
  int *its = ptdata->its;
  int maxit = ptdata->maxit;
  
  int ncells_per_comm = dofi->ncells_per_comm;
  int meps_per_cell = dofi->meps_per_cell;
  int meps_per_comm = dofi->meps_per_comm;
  int meps_total = dofi->meps_total;
  
  PetscScalar *dofext=(PetscScalar *)malloc(meps_total*sizeof(PetscScalar));
  Vec _dofext, _dof;
  MatCreateVecs(W,&_dof,&_dofext);
  array2mpi_f2c(dof,_dof);
  MatMult(W,_dof,_dofext);
  Vec rho_out,rho_grad;
  VecDuplicate(_dofext,&rho_out);
  VecDuplicate(_dofext,&rho_grad);
  threshold_projection_filter(_dofext,rho_out,rho_grad,filter_eta,filter_beta);
  mpi2array_c2c(rho_out,dofext,meps_total);

  PetscScalar uobj_per_comm=0, vobj_per_comm=0, wobj_per_comm=0;
  PetscScalar *ugrad_per_comm = (PetscScalar *)malloc(meps_per_comm*sizeof(PetscScalar));
  PetscScalar *vgrad_per_comm = (PetscScalar *)malloc(meps_per_comm*sizeof(PetscScalar));
  PetscScalar *wgrad_per_comm = (PetscScalar *)malloc(meps_per_comm*sizeof(PetscScalar));
  for(int i=0;i<ncells_per_comm;i++){
    int j=i+ncells_per_comm*colour;

    Vec b;
    VecDuplicate(dg.vecTemp,&b);
    vecfill_zslice(subcomm, dg.da, dofi->mx,dofi->my, Jx[i],Jy[i], NULL, b, iz_src);
    VecScale(b,-PETSC_i*omega);

    if(ux && uy)
      uobj_per_comm+=objcell(subcomm, &(dofext[j*meps_per_cell]), CurlCurl,epsDiff,epsBkg, dofi,dg.da, omega, b, x[i], ksp[i], &(its[i]), maxit, ux[i],uy[i],iz_mtr, &(ugrad_per_comm[i*meps_per_cell]));
    if(vx && vy)
      vobj_per_comm+=objcell(subcomm, &(dofext[j*meps_per_cell]), CurlCurl,epsDiff,epsBkg, dofi,dg.da, omega, b, x[i], ksp[i], &(its[i]), maxit, vx[i],vy[i],iz_mtr, &(vgrad_per_comm[i*meps_per_cell]));
    if(wx && wy)
      wobj_per_comm+=objcell(subcomm, &(dofext[j*meps_per_cell]), CurlCurl,epsDiff,epsBkg, dofi,dg.da, omega, b, x[i], ksp[i], &(its[i]), maxit, wx[i],wy[i],iz_mtr, &(wgrad_per_comm[i*meps_per_cell]));

    VecDestroy(&b);

  }
  
  PetscScalar uobj=0,vobj=0,wobj=0;
  double *ugrad = (double *)malloc(meps_total*sizeof(double));
  double *vgrad = (double *)malloc(meps_total*sizeof(double));
  double *wgrad = (double *)malloc(meps_total*sizeof(double));
  double *mgrad = (double *)malloc(meps_total*sizeof(double));
  if(ux && uy)    
    consolidate(subcomm,colour,meps_total,meps_per_comm,uobj_per_comm,&uobj,ugrad_per_comm,ugrad);
  if(vx && vy)
    consolidate(subcomm,colour,meps_total,meps_per_comm,vobj_per_comm,&vobj,vgrad_per_comm,vgrad);
  if(wx && wy)
    consolidate(subcomm,colour,meps_total,meps_per_comm,wobj_per_comm,&wobj,wgrad_per_comm,wgrad);
  for(int i=0;i<meps_total;i++){
    mgrad[i]=0;
    if(ux && uy)
      mgrad[i]+=ugrad[i];
    if(vx && vy)
      mgrad[i]+=vgrad[i];
    if(wx && wy)
      mgrad[i]+=wgrad[i];
  }
  
  array2mpi_f2c(mgrad,rho_out);
  VecPointwiseMult(rho_out,rho_out,rho_grad);
  MatMultTranspose(W,rho_out,_dof);
  mpi2array_c2f(_dof,grad,ndof);

  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  free(dofext);
  free(ugrad_per_comm);
  free(vgrad_per_comm);
  free(wgrad_per_comm);
  free(ugrad);
  free(vgrad);
  free(wgrad);
  free(mgrad);
  VecDestroy(&_dof);
  VecDestroy(&_dofext);
  VecDestroy(&rho_out);
  VecDestroy(&rho_grad);

  double obj_total=pow(cabs(uobj),2) + pow(cabs(vobj),2) + pow(cabs(wobj),2);

  PetscPrintf(PETSC_COMM_WORLD,"objval at step %d for specID %d is %.16g \n",count,specID,obj_total);

  int print_at = ptdata->print_at_singleobj;
  if(print_at>0 && (count%print_at)==0){
    char output_filename[PETSC_MAX_PATH_LEN];
    sprintf(output_filename,"outputdof_step%d.txt",count);
    writetofile_f2f(PETSC_COMM_WORLD,output_filename,dof,ndof);
  }
  
  count++;
  
  return obj_total;
  
  
}

#undef __FUNCT__
#define __FUNCT__ "ffintensity_maximinconstraint"
double ffintensity_maximinconstraint(int ndof_with_dummy, double *dof_with_dummy, double *grad_with_dummy, void *data)
{
  int ndof=ndof_with_dummy-1;
  double obj=ffintensity(ndof,&(dof_with_dummy[0]),&(grad_with_dummy[0]),data);

  for(int i=0;i<ndof;i++){
    grad_with_dummy[i]=-1.0*grad_with_dummy[i];
  }
  grad_with_dummy[ndof]=1.0;

  count--;

  return dof_with_dummy[ndof]-obj;

}
