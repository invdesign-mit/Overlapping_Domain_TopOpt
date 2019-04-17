import numpy as np
import h5py as hp

filename='focus2d-farfield-freq1.0-angle0.h5'
outputfilename='meepff_spec0'
h5out=0

fid=hp.File(filename,"r")
Effx=np.array(fid["ex.r"])+1j*np.array(fid["ex.i"])
Effy=np.array(fid["ey.r"])+1j*np.array(fid["ey.i"])
Effz=np.array(fid["ez.r"])+1j*np.array(fid["ez.i"])
Eff=np.absolute(Effx)*np.absolute(Effx) + np.absolute(Effy)*np.absolute(Effy) + np.absolute(Effz)*np.absolute(Effz)
Eff=Eff/np.max(Eff.flatten())
fid.close()

if h5out==1:
    fid=hp.File(outputfilename+'.h5','w')
    fid.create_dataset("data",data=Eff)
    fid.close()
else:
    np.savetxt(outputfilename+'.dat',Eff)
