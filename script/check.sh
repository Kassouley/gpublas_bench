#!/bin/bash

############################################################
# USAGE                                                    #
############################################################
usage()
{
usage="
  Usage: $0 {options} <kernel>\n
  options:\n
    \t-h,--help : print help\n
    \t-v,--verbose : print all make output\n
    \t-S,--SP : run simple precision matrix mul\n
    \t-D,--DP : run double precision matrix mul\n
    \t-mXXX : the m dimension of the matrix\n
    \t-kXXX : the k dimension of the matrix\n
  m: default value = 100\n
  k: default value = 100\n
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
    echo -e "gpublas: error in $0\n\t$1 ($err)"
    echo "Script exit."
    exit 1
  fi
}

############################################################
# FUNCTIONS                                                #
############################################################
build_kernel()
{
  eval echo "Build $kernel_lowercase kernel . . . " $output
  eval make check -B  KERNEL=$kernel_uppercase $output
  check_error "make failed"
  eval echo "Done" $output
}

check_kernel()
{
  build_kernel
  output_file="./output/check_$kernel_lowercase.out"
  cmd="./check $m $k $output_file"
  eval echo "exec command : $cmd" $output
  eval $cmd
  check_error "run failed"
  echo "Check kernel $kernel_lowercase : $(python3 ./python/check.py ./output/check_cblas.out $output_file $m)"
  check_error "script python failed"
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

kernel_list=( rocblas rocblas_wo_dt cublas cublas_wo_dt )
kernel_to_check=()
verbose=0
output="> /dev/null"
precision="DP"
all=0
m=100
k=100

TEMP=$(getopt -o haSDvm:k: \
              -l help,all,SP,DP,verbose \
              -n $(basename $0) -- "$@")

if [ $? != 0 ]; then usage ; fi

eval set -- "$TEMP"

while true ; do
    case "$1" in
        -a|--all) kernel_to_check=${kernel_list[@]} ; shift ;;
        -v|--verbose) verbose=1 ; shift ;;
        -m) m=$2 ; shift 2;;
        -k) k=$2 ; shift 2;;
        -S|--SP) precision="SP" ; shift ;;
        -D|--DP) precision="DP" ; shift ;;
        --) shift ; break ;;
        -h|--help) usage ;;
        *) echo "No option $1."; usage ;;
    esac
done

############################################################
# CHECK ARGS                                               #
############################################################
re='^[0-9]+$'
for i in $@; do 
  if [[ " ${kernel_list[*]} " =~ " ${i} " ]] && [ $all == 0 ]; then
    kernel_to_check+=" $i"
  fi
done

if [[ ${kernel_to_check[@]} == "" ]]; then
  echo "No kernel to check"
  exit 1
fi
kernel_to_check=`echo "$kernel_to_check" | tr '[:upper:]' '[:lower:]'`

if [ $verbose == 1 ]; then
  output=""
  echo "Verbose mode on"
fi 

############################################################
# START CHECK                                              #
############################################################
echo "Début des vérifications des sorties"
eval echo -e "Check base kernel . . ." $output
eval make check KERNEL=CBLAS -B $output
check_error "make echoué"
eval ./check $m $k "./output/check_cblas.out"
check_error "lancement du programme echoué"

for i in $kernel_to_check; do
  kernel_lowercase=`echo "$i" | tr '[:upper:]' '[:lower:]'`
  kernel_uppercase=`echo "$i" | tr '[:lower:]' '[:upper:]'`
  if [[ " ${kernel_list[*]} " =~ " ${kernel_lowercase} " ]]; then
    check_kernel
  fi
done

eval make clean $output
echo "Vérifications terminées"
