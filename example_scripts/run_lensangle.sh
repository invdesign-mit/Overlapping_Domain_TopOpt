np=$1
config_file=$2
a0=0.1108
a1=0.1017
a2=0.1036
b0=$(echo "1/sqrt($a0)" | bc -l)
b1=$(echo "1/sqrt($a1)" | bc -l)
b2=$(echo "1/sqrt($a2)" | bc -l)
ffx=57.735026918962575
ffdatout=ffdata2.dat

mpirun -np $np symlens_ffintensity_exec \
       -options_file $config_file \
       -Job 0 \
       -specID 2 \
       -printEfield 0 \
       -is,s0,s1,ds 0,0,1,0.01 \
       -print_at_singleobj 1 \
       -print_at_multiobj 1 \
       -initial_dummy 0 \
       -algouter 24 \
       -alginner 24 \
       -algmaxeval 1000 \
       -ffwindow_cen $ffx,0,100 \
       -ffwindow_size 50,0.4,0.4 \
       -ffwindow_dh 0.1,0.4,0.4 \
       -ffwindow_oxy,symxy 100,0.01,0,0 \
       -ffwindow_outputfilename $ffdatout \
       -spec0_kx,ky,ax,ay 0.0,0.0,0.0,0.0,$b0,0.0 \
       -spec1_kx,ky,ax,ay 2.488098272751806,0.0,0.0,0.0,$b1,0.0 \
       -spec2_kx,ky,ax,ay 4.806636759992383,0.0,0.0,0.0,$b2,0.0 \
       -spec3_kx,ky,ax,ay 0.0,0.0,0.0,0.0,$b0,0.0 \
       -spec4_kx,ky,ax,ay 2.488098272751806,0.0,0.0,0.0,$b1,0.0 \
       -spec5_kx,ky,ax,ay 4.806636759992383,0.0,0.0,0.0,$b2,0.0 \
       -mirrorX 1 \
       -mirrorY 0







