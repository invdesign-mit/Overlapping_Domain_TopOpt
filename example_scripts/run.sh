np=$1
mpirun -np $np lens_ffintensity_exec \
       -options_file config \
       -Job 2 \
       -specID 0 \
       -printEfield 1 \
       -is,s0,s1,ds 0,0,1,0.01 \
       -print_at_singleobj 1 \
       -print_at_multiobj 1 \
       -initial_dummy 0 \
       -algouter 24 \
       -alginner 24 \
       -algmaxeval 1000 \
       -ffwindow_cen 0,0,100 \
       -ffwindow_size 50,0.4,0.4 \
       -ffwindow_dh 0.1,0.4,0.4 \
       -ffwindow_oxy,symxy 100,0.01,0,0 \
       -ffwindow_outputfilename ffdata.dat




