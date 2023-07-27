ifeq ($(CLOCK),)
	CLOCK=MS
endif

ifeq ($(P),)
	P=DP
endif

# -------------------- CC -------------------- #
CC=none
ifeq ($(CC),none)
	ifeq ($(KERNEL), ROCBLAS)
		CC=hipcc
	else ifeq ($(KERNEL), CUBLAS)
		CC=nvcc
	else
		CC=gcc
	endif
endif

# --------------- SRC & FLAGS ---------------- #
CFLAGS=-g -O3 -lm -I./include -D $(P) -D $(KERNEL)
CMEASURE=-D $(CLOCK)

ifeq ($(KERNEL), ROCBLAS)
	LFLAGS=-lrocblas
	SRC_KERNEL=./src/kernel/kernel_rocblas.cpp
	SRC_DRIVER=./src/bench/driver_gpublas.cpp
	SRC_CHECKER=./src/check/driver_check.cpp
else ifeq ($(KERNEL), CUBLAS)
	LFLAGS=-lcublas
	OPT_FLAGS=-gencode=arch=compute_52,code=sm_52 \
			-gencode=arch=compute_60,code=sm_60 \
			-gencode=arch=compute_61,code=sm_61 \
			-gencode=arch=compute_70,code=sm_70 \
			-gencode=arch=compute_75,code=sm_75 \
			-gencode=arch=compute_80,code=sm_80 \
			-gencode=arch=compute_80,code=compute_80
	SRC_KERNEL=./src/kernel/kernel_cublas.cu
	SRC_DRIVER=./src/bench/driver_gpublas.cpp
	SRC_CHECKER=./src/check/driver_check.cpp
else ifeq ($(KERNEL), CBLAS)
	LFLAGS=-lblas
	SRC_DRIVER=./src/bench/driver_cblas.c
	SRC_CHECKER=./src/check/driver_check.c
endif


all: check measure

check: $(SRC_KERNEL) $(SRC_CHECKER) src/tab.c 
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS) $(OPT_FLAGS)

measure: $(SRC_DRIVER) $(SRC_KERNEL) src/tab.c src/rdtsc.c src/print_measure.c src/time_measure.c 
	$(CC) -o $@ $^ $(CFLAGS) $(CMEASURE) $(LFLAGS) $(OPT_FLAGS)

clean:
	rm -rf *.o check calibrate measure