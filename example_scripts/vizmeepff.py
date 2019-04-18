import numpy as np
import h5py as hp

filename=raw_input("Enter the name of the farfield h5 input file from meep: ")
outputfilename=raw_input("Enter the prefix for the name of the output farfield intensity file: ")
h5out=int(raw_input("Generate an h5 output? 0 or 1: "))
normalize=int(raw_input("Normalize the data by the maximum value? 0 or 1: "))

fid=hp.File(filename,"r")
Effx=np.array(fid["ex.r"])+1j*np.array(fid["ex.i"])
Effy=np.array(fid["ey.r"])+1j*np.array(fid["ey.i"])
Effz=np.array(fid["ez.r"])+1j*np.array(fid["ez.i"])
Eff=np.absolute(Effx)*np.absolute(Effx) + np.absolute(Effy)*np.absolute(Effy) + np.absolute(Effz)*np.absolute(Effz)
if normalize==1:
    maxval = np.max(Eff.flatten())
else:
    maxval = 1
Eff=Eff/maxval
fid.close()

if h5out==1:
    fid=hp.File(outputfilename+'.h5','w')
    fid.create_dataset("data",data=Eff)
    fid.close()
else:
    np.savetxt(outputfilename+'.dat',Eff)
