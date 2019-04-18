import numpy as np
import h5py as hp
import os

numcells_x=int(raw_input("Enter the number of cells along X axis: "))
numcells_y=int(raw_input("Enter the number of cells along Y axis: "))

join=int(raw_input("Join the cells and generate the full domain? 0 or 1: "))
printEfield=int(raw_input("Print the Electric fields? 0 or 1: "))

mirrorX=int(raw_input("Mirror the structure along X, eps(x)=eps(-x)? 0 or 1: "))
mirrorY=int(raw_input("Mirror the structure along Y, eps(y)=eps(-y)? 0 or 1: "))

prefix=raw_input("Enter the prefix for the names of the h5 output files: ")

if join==1:
    epsxy=[]
    for icy in range(numcells_y):
        epsx=[]
        for icx in range(numcells_x):
            name='epscell'+str(icx)+'_'+str(icy)+'.h5'
            fid=hp.File(name,'r')
            eps=np.array(fid['eps'])
            eps=eps[:,:,:,0,0]
            fid.close()
            if epsx==[]:
                epsx=eps
            else:
                epsx=np.concatenate((epsx,eps),axis=2)
        if epsxy==[]:
            epsxy=epsx
        else:
            epsxy=np.concatenate((epsxy,epsx),axis=1)
    [nz,ny,nx]=epsxy.shape
    epsmeep=np.zeros((nx,ny,nz))
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                epsmeep[ix,iy,iz]=epsxy[iz,iy,ix]
    if mirrorX==1:
        epsmeep=np.concatenate((np.flip(epsmeep,axis=0),epsmeep),axis=0)
    if mirrorY==1:
        epsmeep=np.concatenate((np.flip(epsmeep,axis=1),epsmeep),axis=1)
    fid=hp.File(prefix+'epsilon.h5','w')
    fid.create_dataset('eps',data=np.squeeze(epsmeep))
    fid.close()
else:
    for icy in range(numcells_y):
        for icx in range(numcells_x):
            name='epscell'+str(icx)+'_'+str(icy)
            fid=hp.File(name+'.h5','r')
            eps=np.array(fid['eps'])
            eps=np.squeeze(eps[:,:,:,0,0])
            fid.close()
            fid=hp.File(prefix+name+'.h5','w')
            fid.create_dataset('eps',data=eps)
            fid.close()


if printEfield==1:
    if join==1:
        ExrXY=[]
        ExiXY=[]
        EyrXY=[]
        EyiXY=[]
        EzrXY=[]
        EziXY=[]
        for icy in range(numcells_y):
            ExrX=[]
            ExiX=[]
            EyrX=[]
            EyiX=[]
            EzrX=[]
            EziX=[]
            for icx in range(numcells_x):
                name='Efield'+str(icx)+'_'+str(icy)
                fid=hp.File(name+'.h5','r')
                E=np.array(fid['E'])
                Exr=E[:,:,:,0,0]
                Exi=E[:,:,:,0,1]
                Eyr=E[:,:,:,1,0]
                Eyi=E[:,:,:,1,1]
                Ezr=E[:,:,:,2,0]
                Ezi=E[:,:,:,2,1]
                if ExrX==[]:
                    ExrX=Exr
                else:
                    ExrX=np.concatenate((ExrX,Exr),axis=2)
                if ExiX==[]:
                    ExiX=Exi
                else:
                    ExiX=np.concatenate((ExiX,Exi),axis=2)
                if EyrX==[]:
                    EyrX=Eyr
                else:
                    EyrX=np.concatenate((EyrX,Eyr),axis=2)
                if EyiX==[]:
                    EyiX=Eyi
                else:
                    EyiX=np.concatenate((EyiX,Eyi),axis=2)
                if EzrX==[]:
                    EzrX=Ezr
                else:
                    EzrX=np.concatenate((EzrX,Ezr),axis=2)
                if EziX==[]:
                    EziX=Ezi
                else:
                    EziX=np.concatenate((EziX,Ezi),axis=2)
            if ExrXY==[]:
                ExrXY=ExrX
            else:
                ExrXY=np.concatenate((ExrXY,ExrX),axis=1)
            if ExiXY==[]:
                ExiXY=ExiX
            else:
                ExiXY=np.concatenate((ExiXY,ExiX),axis=1)
            if EyrXY==[]:
                EyrXY=EyrX
            else:
                EyrXY=np.concatenate((EyrXY,EyrX),axis=1)
            if EyiXY==[]:
                EyiXY=EyiX
            else:
                EyiXY=np.concatenate((EyiXY,EyiX),axis=1)
            if EzrXY==[]:
                EzrXY=EzrX
            else:
                EzrXY=np.concatenate((EzrXY,EzrX),axis=1)
            if EziXY==[]:
                EziXY=EziX
            else:
                EziXY=np.concatenate((EziXY,EziX),axis=1)
        fid=hp.File(prefix+'Efield.h5','w')
        fid.create_dataset('Exr',data=ExrXY)
        fid.create_dataset('Exi',data=ExiXY)
        fid.create_dataset('Eyr',data=EyrXY)
        fid.create_dataset('Eyi',data=EyiXY)
        fid.create_dataset('Ezr',data=EzrXY)
        fid.create_dataset('Ezi',data=EzrXY)
        fid.close()
    else:
        for icy in range(numcells_y):
            for icx in range(numcells_x):
                name='Efield'+str(icx)+'_'+str(icy)
                fid=hp.File(name,'r')
                E=np.array(fid['E'])
                fid.close()
                fid=hp.File(prefix+name+'.h5','w')
                fid.create_dataset('Exr',data=np.squeeze(E[:,:,:,0,0]))
                fid.create_dataset('Exi',data=np.squeeze(E[:,:,:,0,1]))
                fid.create_dataset('Eyr',data=np.squeeze(E[:,:,:,1,0]))
                fid.create_dataset('Eyi',data=np.squeeze(E[:,:,:,1,1]))
                fid.create_dataset('Ezr',data=np.squeeze(E[:,:,:,2,0]))
                fid.create_dataset('Ezi',data=np.squeeze(E[:,:,:,2,1]))
                fid.close()
