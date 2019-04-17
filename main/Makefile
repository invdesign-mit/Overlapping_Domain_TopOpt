export PETSC_DIR=${HOME}/petsc-3.6.4
export PETSC_ARCH=arch-mumps-opt
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CLEANFILES = *.o *_exec

CC=mpicc

FDOPT_INC = -I ../library/
FDOPT_LIB = ../library/libFDOPT.a

NLOPT_INC = -I /usr/local/include
NLOPT_LIB = /usr/local/lib/libnlopt.a

CFLAGS   += -O3 -Wall -march=native -fcx-limited-range -fno-exceptions
INCFLAGS = -I. ${FDOPT_INC} ${NLOPT_INC} ${PETSC_CC_INCLUDES} 
LIBS=$(FDOPT_LIB) $(NLOPT_LIB) $(PETSC_LIB)

all: steer_ffintensity_exec lens_ffintensity_exec

steer_ffintensity.o: steer_ffintensity.c
	$(CC) -c $(CFLAGS) $(INCFLAGS) $< -o $@
steer_ffintensity_exec: steer_ffintensity.o
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

lens_ffintensity.o: lens_ffintensity.c
	$(CC) -c $(CFLAGS) $(INCFLAGS) $< -o $@
lens_ffintensity_exec: lens_ffintensity.o
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
