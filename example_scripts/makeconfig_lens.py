import numpy as np
import h5py as hp
import os
from scipy.interpolate import interp1d

################################
def writeflag(fid,pre,x):
    if hasattr(x, "__len__" ):
        strx=pre+" "+str(x[0])
        for y in x[1::]:
            strx+=","+str(y)
    else:
        strx=pre+" "+str(x)
    strx+="\n"
    fid.write(strx)
    return strx

def mat_sio2(lamnm):
    lam=lamnm/1000.0
    epsilon=1 + 0.6961663*lam**2/(lam**2-0.0684043**2) + 0.4079426*lam**2/(lam**2-0.1162414**2) + 0.8974794*lam**2/(lam**2-9.896161**2)
    return epsilon

def mat_tio2(lamnm):
    
    file='tio2_refractiveindex_[nm].txt'
    data=np.loadtxt(file)

    n=interp1d(data[:,0],data[:,1])(lamnm)
    eps=n**2
    return eps

class grid:
    def __init__(self, bc,xraw,yraw,zraw,mpml,ekl):
        self.bc=bc
        self.xraw=xraw
        self.yraw=yraw
        self.zraw=zraw
        self.mpml=mpml
        self.ekl=ekl

    def printgrid(self,name):
        fid=hp.File(name,"w")
        fid.create_dataset("bc",data=self.bc)
        fid.create_dataset("xraw",data=self.xraw)
        fid.create_dataset("yraw",data=self.yraw)
        fid.create_dataset("zraw",data=self.zraw)
        fid.create_dataset("Mpml",data=self.mpml)
        fid.create_dataset("e_ikL",data=self.ekl)
        fid.close()
                                                                                                                                               

###############################

numcells_x=10
numcells_y=1
ncells_per_comm=2
nproc_per_comm=10
print "******* IMPORTANT Choose np = " + str(numcells_x*numcells_y/ncells_per_comm * nproc_per_comm)

nx=1000
ny=1
numlayers=3

dx=0.02
dy=0.02
dz=0.02

#alternating patterened and uniform layers
mz=[35,35,35]
mid=[5,5]
mpmlz=[25,25]
pml2src=2
src2stk=10
stk2ref=10
ref2pml=70
stk=sum(mz)+sum(mid)
Mz=mpmlz[0]+pml2src+src2stk+stk+stk2ref+ref2pml+mpmlz[1]
mzslab=1
mzo=np.zeros(numlayers,dtype=int)
mzo[0]=mpmlz[0]+pml2src+src2stk
for i in range(numlayers-1):
    mzo[i+1]=mzo[i]+mz[i]+mid[i]
    
bcx=2
bcy=2
bcz=2

nspecs=3

pxymo=np.zeros((4,nspecs),dtype=int)
pxymo[:,0]=[0,0,0,0]
pxymo[:,1]=[100,0,0,0]
pxymo[:,2]=[100,0,0,0]

mpmlx=np.zeros((2,nspecs))
mpmlx[:,0]=[0,0]
mpmlx[:,1]=[0,0]
mpmlx[:,2]=[25,25]
mpmly=np.zeros((2,nspecs))
mpmly[:,0]=[0,0]
mpmly[:,1]=[0,0]
mpmly[:,2]=[0,0]

freq=[1,1,1]

nsub=[1.5,1.5,1.5]
nsup=[1,1,1]

#eps shape should be nspecs x 2*total_layers; for each spec, [eps_foreground,eps_background]
#tot_layers=2*nlayers+1
eps=np.array([ ( nsub[i]**2,nsub[i]**2,
                 2.25,1,
                 2.25,2.25,
                 2.25,1,
                 2.25,2.25,
                 2.25,1,
                 nsup[i]**2,nsup[i]**2 ) for i in range(nspecs) ])


polar=[0,0,0]
azimuth=[0,0,0]
ax=np.zeros((2,nspecs))
ay=np.zeros((2,nspecs))
ax[:,0]=[0,0]
ay[:,0]=[1,0]
ax[:,1]=[0,0]
ay[:,1]=[1,0]
ax[:,2]=[0,0]
ay[:,2]=[1,0]
kz=np.zeros(nspecs)

oxy=[nx*numcells_x*dx/2.0,ny*numcells_y*dy/2.0]
symxy=[0,0]
xyzfar=np.zeros((3,nspecs))
xyzfar[:,0]=[0,0,100]
xyzfar[:,1]=[0,0,100]
xyzfar[:,2]=[0,0,100]

filter_rx=1
filter_ry=1
filter_alpha=5
filter_normalized=1
filter_beta=0
filter_eta=0.3

init_filename='dof.txt'

chkeps=1

#define periodic dof by single cell
dof=np.zeros(ny*nx*numlayers)
for iy in range(ny):
    for ix in range(nx):
        for il in range(numlayers):
            i=il + numlayers*ix + numlayers*nx*iy
            if (ix-nx/2)**2 + (iy-ny/2)**2 <= 1000**2:
                #dof[i]=round(np.random.rand())
                #dof[i]=np.random.rand()
                dof[i]=0.5
fid=hp.File('dof_singlecell.h5','w')
tmp=dof.reshape((ny,nx,numlayers))
for il in range(numlayers):
    fid.create_dataset('layer'+str(il),data=np.squeeze(tmp[:,:,il]))
fid.close()
tmp=dof
dof=np.array([])
for icy in range(numcells_y):
    for icx in range(numcells_x):
        if dof.size==0:
            dof=tmp
        else:
            dof=np.concatenate((dof,tmp))
np.savetxt(init_filename,dof)

#############################################################################################################

iz=[0,mpmlz[0]+pml2src+src2stk]
for i in range(numlayers-1):
    tmp=[iz[-1],iz[-1]+mz[i]]
    iz=np.concatenate((iz,tmp))
    tmp=[iz[-1],iz[-1]+mid[i]]
    iz=np.concatenate((iz,tmp))
tmp=[iz[-1],iz[-1]+mz[numlayers-1]]
iz=np.concatenate((iz,tmp))
tmp=[iz[-1],iz[-1]+stk2ref+ref2pml+mpmlz[1]]
iz=np.concatenate((iz,tmp))
print iz

for j in range(nspecs):
    px=pxymo[0,j]
    py=pxymo[1,j]
    mx=nx+2*px
    my=ny+2*py
    epsDiff=np.zeros((Mz,my,mx,3,2))
    epsBkg=np.zeros((Mz,my,mx,3,2))
    for i in range(2*numlayers+1):
        epsdiff=eps[j,2*i+0]-eps[j,2*i+1]
        epsbkg=eps[j,2*i+1]
        tmp = np.array([epsdiff, epsbkg])
        print tmp
        epsDiff[iz[2*i+0]:iz[2*i+1],:,:,:,0]=epsdiff
        epsBkg[iz[2*i+0]:iz[2*i+1],:,:,:,0]=epsbkg
    fid=hp.File('epsDiff'+str(j)+'.h5','w')
    fid.create_dataset('eps',data=epsDiff)
    fid.close()
    fid=hp.File('epsBkg'+str(j)+'.h5','w')
    fid.create_dataset('eps',data=epsBkg)
    fid.close()
    if(chkeps):
        fid=hp.File('chk_epsDiff'+str(j)+'.h5','w')
        fid.create_dataset('eps',data=np.squeeze(epsDiff[:,:,:,0,0]))
        fid.close()
        fid=hp.File('chk_epsBkg'+str(j)+'.h5','w')
        fid.create_dataset('eps',data=np.squeeze(epsBkg[:,:,:,0,0]))
        fid.close()
        
kx=[ 2*np.pi*freq[i]*nsub[i]*np.sin(polar[i]*np.pi/180)*np.cos(azimuth[i]*np.pi/180) for i in range(nspecs) ]
ky=[ 2*np.pi*freq[i]*nsub[i]*np.sin(polar[i]*np.pi/180)*np.sin(azimuth[i]*np.pi/180) for i in range(nspecs) ]

for i in range(nspecs):
    px=pxymo[0,i]
    py=pxymo[1,i]
    mx=nx+2*px
    my=ny+2*py
    Lx=mx*dx
    Ly=my*dy
    Lz=Mz*dz
    x_start=0
    y_start=0
    z_start=0
    x_end=x_start+Lx+dx
    y_end=y_start+Ly+dy
    z_end=z_start+Lz+dz
    xraw=np.arange(x_start,x_end,dx)
    yraw=np.arange(y_start,y_end,dy)
    zraw=np.arange(z_start,z_end,dz)
    if xraw.size-1!=mx:
        print "ERROR: xraw.size not consistent with #pixels along x\n"
    if yraw.size-1!=my:
        print "ERROR: yraw.size not consistent with #pixels along y\n"
    if zraw.size-1!=Mz:
        print "ERROR: zraw.size not consistent with #pixels along z\n"
    bc=np.array((bcx,bcy,bcz))
    mpml=np.array((mpmlx[0,i],mpmlx[1,i],mpmly[0,i],mpmly[1,i],mpmlz[0],mpmlz[1])).reshape(3,2)
    eklx=np.exp(-1j*kx[i]*Lx)
    ekly=np.exp(-1j*ky[i]*Ly)
    eklz=np.exp(-1j*kz[i]*Lz)
    ekl=np.array((eklx.real,eklx.imag,ekly.real,ekly.imag,eklz.real,eklz.imag)).reshape(3,2)
    gi=grid(bc,xraw,yraw,zraw,mpml,ekl)
    gi.printgrid("grid"+str(i)+".h5")

fid=open("config","w")

writeflag(fid,"-numcells_x",numcells_x)
writeflag(fid,"-numcells_y",numcells_y)
writeflag(fid,"-ncells_per_comm",ncells_per_comm)
writeflag(fid,"-nproc_per_comm",nproc_per_comm)
writeflag(fid,"-nx",nx)
writeflag(fid,"-ny",ny)
writeflag(fid,"-numlayers",numlayers)
writeflag(fid,"-mz",mz)
writeflag(fid,"-mzo",mzo)
writeflag(fid,"-iz_source",mpmlz[0]+pml2src)
writeflag(fid,"-iz_monitor",mpmlz[0]+pml2src+src2stk+stk+stk2ref)
writeflag(fid,"-dx",dx)
writeflag(fid,"-dy",dy)
writeflag(fid,"-dz",dz)
writeflag(fid,"-mzslab",mzslab)
writeflag(fid,"-filter_rx",filter_rx)
writeflag(fid,"-filter_ry",filter_ry)
writeflag(fid,"-filter_alpha",filter_alpha)
writeflag(fid,"-filter_normalized",filter_normalized)
writeflag(fid,"-filter_beta",filter_beta)
writeflag(fid,"-filter_eta",filter_eta)
writeflag(fid,"-nspecs",nspecs)
fid.write("-init_dof_name "+init_filename+"\n")

for i in range(nspecs):
    writeflag(fid,"-spec"+str(i)+"_freq",freq[i])
    writeflag(fid,"-spec"+str(i)+"_px,py,mxo,myo",np.squeeze(pxymo[:,i]))
    writeflag(fid,"-spec"+str(i)+"_kx,ky,ax,ay",[kx[i],ky[i],ax[0,i],ax[1,i],ay[0,i],ay[1,i]])
    writeflag(fid,"-spec"+str(i)+"_out_oxy,symxy,xyzfar",[oxy[0],oxy[1],symxy[0],symxy[1],xyzfar[0,i],xyzfar[1,i],xyzfar[2,i]])
fid.close()

