DEBUG=-Wall -g -DDEBUG
LDFLAGS=-g -lm -ljpeg -O3
BINSGPU=julia_gpu
BINSCPU=julia_cpu
CC=gcc
NVCC=nvcc
FILECPU=main_cpu.c
FILEGPU=main_gpu.cu
all: ${BINS}

cpu: clean_cpu
		${CC} ${FILECPU} ${LDFLAGS} -o ${BINSCPU}

gpu: clean_gpu
		${NVCC} ${FILEGPU} ${LDFLAGS} -o ${BINSGPU}



clean_gpu:
			/bin/rm -rf ${BINSGPU} *.o core *.core

clean_cpu:
			/bin/rm -rf ${BINSCPU} *.o core *.core
