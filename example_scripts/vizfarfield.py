import numpy as np
import h5py as hp

ffnx=500
ffny=1
ffnz=1

fname='ffdata.dat'
h5name='ffdata.h5'

h5out=0

data=np.loadtxt(fname)
maxdat=np.max(data)
data=data/maxdat
if h5out==1:

    h5dat=np.zeros((ffnx,ffny,ffnz))
    for iz in range(ffnz):
        for iy in range(ffny):
            for ix in range(ffnx):
                i=ix+ffnx*iy+ffnx*ffny*iz
                h5dat[ix,iy,iz]=data[i]

    fid=hp.File(h5name,'w')
    fid.create_dataset('data',data=h5dat)
    fid.close()

else:
    np.savetxt(fname,data)
