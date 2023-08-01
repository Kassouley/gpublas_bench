ifeq ($(P),)
	P=DP
endif

IS_KERNEL_ROCBLAS := $(filter $(KERNEL), ROCBLAS ROCBLAS_WO_DT)
IS_KERNEL_CUDA := $(filter $(KERNEL), CUBLAS CUBLAS_WO_DT)

# -------------------- CC -------------------- #
CC=none
ifeq ($(CC),none)
	ifneq ($(IS_KERNEL_ROCBLAS),)
		CC=hipcc
	else ifneq ($(IS_KERNEL_CUDA),)
		CC=nvcc
	else
		CC=gcc
	endif
endif

# --------------- SRC & FLAGS ---------------- #
CFLAGS=-g -O3 -lm -I./include -D $(P) -D $(KERNEL) -lm

IS_KERNEL_IN_CPP := $(filter $(KERNEL), ROCBLAS CUBLAS)
IS_KERNEL_IN_CPP_WO_DT := $(filter $(KERNEL), ROCBLAS_WO_DT CUBLAS_WO_DT)

ifneq ($(IS_KERNEL_ROCBLAS),)
	LFLAGS=-lrocblas -fopenmp
	SRC_KERNEL=./src/kernel/kernel_rocblas.cpp
else ifneq ($(IS_KERNEL_CUDA),)
	LFLAGS=-lcublas -Xcompiler -fopenmp
	OPT_FLAGS=-gencode=arch=compute_52,code=sm_52 \
			-gencode=arch=compute_60,code=sm_60 \
			-gencode=arch=compute_61,code=sm_61 \
			-gencode=arch=compute_70,code=sm_70 \
			-gencode=arch=compute_75,code=sm_75 \
			-gencode=arch=compute_80,code=sm_80 \
			-gencode=arch=compute_80,code=compute_80
	SRC_KERNEL=./src/kernel/kernel_cublas.cu
else ifeq ($(KERNEL), CBLAS)
	LFLAGS=-lblas -fopenmp
	SRC_DRIVER=./src/bench/driver_cblas.c
	SRC_CHECKER=./src/check/driver_check.c
endif

ifneq ($(IS_KERNEL_IN_CPP_WO_DT),)
	SRC_DRIVER=./src/bench/driver_gpublas_wo_dt.cpp
	SRC_CHECKER=./src/check/driver_check_wo_dt.cpp
else ifneq ($(IS_KERNEL_IN_CPP),)
	SRC_DRIVER=./src/bench/driver_gpublas.cpp
	SRC_CHECKER=./src/check/driver_check.cpp
endif


all: check measure

check: $(SRC_KERNEL) $(SRC_CHECKER) src/tab.c 
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS) $(OPT_FLAGS)

measure: $(SRC_DRIVER) $(SRC_KERNEL) src/tab.c src/print_measure.c 
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS) $(OPT_FLAGS)

clean:
	rm -rf *.o check calibrate measure
