import numpy as np
import h5py as hp

ffnx=int(raw_input("Enter # of points along x: "))
ffny=int(raw_input("Enter # of points along y: "))
ffnz=int(raw_input("Enter # of points along z: "))

fname=raw_input("Enter the name of the input file: ")
prefix_output=raw_input("Enter the prefix for the name of the output file: ")

h5out=int(raw_input("Generate an h5 output? 0 or 1: "))
normalize=int(raw_input("Normalize the data by maximum value? 0 or 1: "))

data=np.loadtxt(fname)
if normalize==1:
    maxdat=np.max(data)
else:
    maxdat=1
data=data/maxdat

if h5out==1:

    h5dat=np.zeros((ffnx,ffny,ffnz))
    for iz in range(ffnz):
        for iy in range(ffny):
            for ix in range(ffnx):
                i=ix+ffnx*iy+ffnx*ffny*iz
                h5dat[ix,iy,iz]=data[i]

    fid=hp.File(prefix_output+'.h5','w')
    fid.create_dataset('data',data=h5dat)
    fid.close()

else:
    np.savetxt(prefix_output+'.dat',data)
