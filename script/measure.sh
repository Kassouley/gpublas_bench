#!/bin/bash

############################################################
# USAGE                                                    #
############################################################
usage()
{
  usage="
  \nUsage: $0 {options} [problem size] <kernels>\n\n
  options:\n
    \t-h,--help : print help\n
    \t-a,-all : measure all kernel\n
    \t-p,--plot={plot_file} : create a png plot with the results in png file in argument (default: ./graphs/graph_DATE.png)\n
    \t-s,--save={out_file} : save the measure output in file in argument (default: ./output/measure_DATE.out)\n
    \t-v,--verbose : print all make output\n
    \t-m,--milliseconds : time is in milliseconds instead of RTDSC-Cycles\n
    \t-f,--force : do not ask for starting the measure\n
  problem size :\n\tdefault value = 100\n
  kernels:\n
    \t${kernel_list[*]}\n"

  echo -e $usage
  exit 1
}

############################################################
# IF ERROR                                                 #
############################################################

check_error()
{
  err=$?
  if [ $err -ne 0 ]; then
    echo -e "gpumm: error in $0\n\t$1 ($err)"
    echo "Script exit."
    exit 1
  fi
}

############################################################
# FUNCTION                                                 #
############################################################
measure_kernel()
{
  echo -e "Measure kernel $KERNEL (A["$1"x"$2"] * B["$2"x"$1"]) . . ."
  cmd="$WORKDIR/measure $1 $2 100 500"
  eval echo "exec command : $cmd" $output
  eval $cmd
  check_error "run measure failed"
}

build_kernel()
{
  echo -e -n "Build kernel $KERNEL . . . "
  eval make measure -B KERNEL=$KERNEL CLOCK=$clock P=$precision $output
  check_error "build failed"
  echo "Done"
}

############################################################
# SETUP                                                    #
############################################################

WORKDIR=`realpath $(dirname $0)/..`
cd $WORKDIR

############################################################
# CHECK GPU                                                #
############################################################
GPU_CHECK=$( lshw -C display 2> /dev/null | grep nvidia )
GPU=NVIDIA
KERNEL=CUBLAS
if [[ -z "$GPU_CHECK" ]]; then
  GPU_CHECK=$( lshw -C display 2> /dev/null | grep amd )
  GPU=AMD
  KERNEL=ROCBLAS
fi
if [[ -z "$GPU_CHECK" ]]; then
  echo "No GPU found."
  exit 1
fi

############################################################
# CHECK OPTIONS                                            #
############################################################

precision="DP"
verbose=0
output="> /dev/null"
force=0
clock="RDTSC"
clock_label="RDTSC-cycles"
save=0
save_file="$WORKDIR/output/measure_$(date +%F-%T).out"

TEMP=$(getopt -o hfSPvms:: \
              -l help,force,SP,DP,verbose,millisecond,save:: \
              -n $(basename $0) -- "$@")

eval set -- "$TEMP"

if [ $? != 0 ]; then usage ; fi

while true ; do
    case "$1" in
        -S|--SP) precision="SP" ; shift ;;
        -D|--DP) precision="DP" ; shift ;;
        -f|--force) force=1 ; shift ;;
        -v|--verbose) verbose=1 ; shift ;;
        -m|--millisecond) clock="MS" ; clock_label="Time (ms)"; shift ;;
        -s|--save) 
            case "$2" in
                "") save=1; shift 2 ;;
                *)  save=1; save_file="$2" ; shift 2 ;;
            esac ;;
        -p|--plot) 
            case "$2" in
                "") plot=1; shift 2 ;;
                *)  plot=1; plot_file="$2" ; shift 2 ;;
            esac ;;
        --) shift ; break ;;
        -h|--help) usage ;;
        *) echo "No option $1."; usage ;;
    esac
done

############################################################
# SUMMARY OF THE RUN                                       #
############################################################
echo -n "Summary measure ($clock_label) on a $precision matrix on $GPU GPU"
if [ $verbose == 1 ]; then
  output=""
  echo -n " (with verbose mode)"
fi 
echo -e "\nKernel to measure : $KERNEL"
if [ $save == 1 ]; then
  echo "Measure will be saved in '$save_file'"
fi 

if [ $force == 0 ]; then
  echo -n "Starting ?"
  while true; do
          read -p " (y/n) " yn
        case $yn in
            [Yy]*) break ;;
            [Nn]*) echo "Aborted" ; exit 1 ;;
        esac
    done
fi

############################################################
# START MEASURE                                            #
############################################################
if [[ -f $WORKDIR/output/measure_tmp.out ]]; then
  rm $WORKDIR/output/measure_tmp.out
fi
echo "   m   |   k   |     minimum     |     median     |   stability" > $WORKDIR/output/measure_tmp.out


build_kernel
k=1
for m in {500..3000..100}
do
    measure_kernel $m $k
    k=$(($k+1))
done

eval make clean $output

if [ $plot == 1 ]; then
  echo "Graph generation . . ."
  python3 ./python/graph-gen-measure.py $data_size $WORKDIR/output/measure_tmp.out $plot_file $clock_label
  echo "Graph created in file $plot_file"
fi

echo "---------------------"
echo "Result Summary : "
cat $WORKDIR/output/measure_tmp.out
echo "---------------------"

if [ $save == 1 ]; then
  mv $WORKDIR/output/measure_tmp.out $save_file
fi

