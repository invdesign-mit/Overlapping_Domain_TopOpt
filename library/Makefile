export PETSC_DIR=${HOME}/petsc-3.6.4
export PETSC_ARCH=arch-mumps-opt
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CLEANFILES = *.o

NLOPT_INC = -I /usr/local/include

CC=mpicc
CFLAGS   += -O3 -Wall -march=native -fcx-limited-range -fno-exceptions
INCFLAGS = -I. ${NLOPT_INC} ${PETSC_CC_INCLUDES}

LIBS4OBJS=ovmat.o array2vec.o input.o output.o planewave.o grid.o dof2dom.o logging.o pml.o vec.o mat.o solver.o filters.o ffintensity.o optitemp.o near2far.o

lib: libFDOPT.a $(LIBS4OBJS)

libFDOPT.a: $(LIBS4OBJS)
	$(AR) crvs $@ *.o


