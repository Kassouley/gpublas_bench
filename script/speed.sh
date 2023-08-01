
WORKDIR=`realpath $(dirname $0)/..`
cd $WORKDIR

python3 ./python/graph-gen-measure.py $WORKDIR/output/measure_rocblas_DP_* rocblas_DP "rocBLAS (DP)"
python3 ./python/graph-gen-measure.py $WORKDIR/output/measure_rocblas_SP_* rocblas_SP "rocBLAS (SP)"
python3 ./python/graph-gen-measure.py $WORKDIR/output/measure_rocblas_wo_dt_DP_* rocblas_wo_dt_DP "rocBLAS (w/o DT) (DP)"
python3 ./python/graph-gen-measure.py $WORKDIR/output/measure_rocblas_wo_dt_SP_* rocblas_wo_dt_SP "rocBLAS (w/o DT) (SP)"

python3 ./python/graph-gen-measure.py $WORKDIR/output/measure_cublas_DP_* cublas_DP "cuBLAS (DP)"
python3 ./python/graph-gen-measure.py $WORKDIR/output/measure_cublas_SP_* cublas_SP "cuBLAS (SP)"
python3 ./python/graph-gen-measure.py $WORKDIR/output/measure_cublas_wo_dt_DP_* cublas_wo_dt_DP "cuBLAS (w/o DT) (DP)"
python3 ./python/graph-gen-measure.py $WORKDIR/output/measure_cublas_wo_dt_SP_* cublas_wo_dt_SP "cuBLAS (w/o DT) (SP)"