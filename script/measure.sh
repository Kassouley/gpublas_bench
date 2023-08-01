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
    \t-p,--plot : create a png plot with the results in a png file \n
    \t-s,--save : save the measure output in an output file\n
    \t-v,--verbose : print all make output\n
    \t-S,--SP : run simple precision matrix mul\n
    \t-D,--DP : run double precision matrix mul\n
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
  echo -e "Measure kernel $kernel (A["$1"x"$2"] * B["$2"x"$1"]) . . ."
  cmd="$WORKDIR/measure $1 $2 300 20"
  eval echo "exec command : $cmd" $output
  eval $cmd
  check_error "run measure failed"
}

build_kernel()
{
  echo -e -n "Build kernel $kernel . . . "
  eval make measure -B KERNEL=$KERNEL P=$precision $output
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
if [[ -z "$GPU_CHECK" ]]; then
  GPU_CHECK=$( lshw -C display 2> /dev/null | grep amd )
  GPU=AMD
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
plot=0
save=0

TEMP=$(getopt -o hfSDvsp \
              -l help,force,SP,DP,verbose,save,plot \
              -n $(basename $0) -- "$@")

eval set -- "$TEMP"

if [ $? != 0 ]; then usage ; fi

while true ; do
    case "$1" in
        -S|--SP) precision="SP" ; shift ;;
        -D|--DP) precision="DP" ; shift ;;
        -f|--force) force=1 ; shift ;;
        -v|--verbose) verbose=1 ; shift ;;
        -s|--save) save=1; shift ;;
        -p|--plot) plot=1; shift ;;
        --) shift ; break ;;
        -h|--help) usage ;;
        *) echo "No option $1."; usage ;;
    esac
done

############################################################
# SUMMARY OF THE RUN                                       #
############################################################
echo -n "Summary measure ($precision matrix) on $GPU GPU"
if [ $verbose == 1 ]; then
  output=""
  echo -n " (with verbose mode)"
fi 
echo -e "\nKernel to measure : $@"
if [ $plot == 1 ]; then
  echo "Plot will be generated in graphs/ dir"
fi 
if [ $save == 1 ]; then
  echo "Measure will be saved in output/ dir"
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

for kernel in $@
do

KERNEL=`echo "$kernel" | tr '[:lower:]' '[:upper:]'`

if [[ -f $WORKDIR/output/measure_tmp.out ]]; then
  rm $WORKDIR/output/measure_tmp.out
fi
echo "   m   |   k   |    GFLOPS/S     |   minimum (ms)   |   median (ms)   |   stability (%)" > $WORKDIR/output/measure_tmp.out

  build_kernel
  for k in {1..32}
  do
    for m in {500..3000..100}
    do
        measure_kernel $m $k
    done
  done

  plot_file="$WORKDIR/graphs/graph_"$kernel"_"$precision"_$(date +%F-%T).png"
  save_file="$WORKDIR/output/measure_"$kernel"_"$precision"_$(date +%F-%T).out"

  if [ $plot == 1 ]; then
    echo "Graph generation . . ."
    python3 ./python/graph-gen-measure.py $WORKDIR/output/measure_tmp.out $plot_file "$kernel ($precision)"
    echo "Graph created in file $plot_file"
  fi

  echo "---------------------"
  echo "Result Summary : "
  cat $WORKDIR/output/measure_tmp.out
  echo "---------------------"

  if [ $save == 1 ]; then
    mv $WORKDIR/output/measure_tmp.out $save_file
  fi
done
eval make clean $output

