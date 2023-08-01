# gpublas_bench

Benchmark of cuBLAS and rocBLAS matrix multiplcation for a (mxk) * (kxm) matrix

## Author

- [@Kass](https://www.github.com/Kassouley) 

![Kass](https://cdn.discordapp.com/attachments/705826516520665191/1116698582557397062/canvas100.png)


## Installation

Advice : use the script in the 'script/' folder

No particulary installation needed.
Just build with :
```bash
make measure KERNEL=[KERNEL_NAME] P=[SP|DP] GPU=[NVIDIA|AMD] CC=whateveruwant
make check KERNEL=[KERNEL_NAME] P=[SP|DP] GPU=[NVIDIA|AMD] CC=whateveruwant
```

KERNEL_NAME should be in uppercase.

GPU is optional (AMD by default)

P is the precision of the matrix multiplication : simple or double precision (DP by default)

CC is optional (AMD Clang on AMD and NVC on NVIDIA (NVCC for cuBLAS kernels) by default)

Then run with :
```bash
./measure <m> <k> <nb warmup> <nb rep>
./check <m> <k> [file name]
```

- m and k are the dim of the (mxk) * (kxm) matrix multiplication
- nb warmup is the number of warmup before starting the bench
- nb rep is the number of repetitions to dampen the accuracy of the timer
- file name is an outfile
    
## Code Features

- Shows us the performance of a kernel in GFLOPS/s and ms
- Benchmark on GPU using rocBLAS with and without the data transfer (on AMD)
- Benchmark on GPU using cuBLAS with and without the data transfer (on NVIDIA)
- Checker for these kernels

## Script Features

### Bash script

measure.sh :
- Perform matrix product benchmarking with k ranging from 1 to 32 and m ranging from 500 to 3000 with increments of 100.
- Output can be saved in a text file
- Can generate a graph based on benchmark outputs

check.sh :
- Check all kernel in arguments for a dimension given
- Automatically compares kernels outputs with the cblas matrix multiplication kernel
- Shows us the error value between two kernel output

### Python script

check.py :
- take in argument two output file and a matrix size
- Check if the two output files are the same and if not, give the max error between these two files

graph-gen-measure.py :
- take in argument an output file from the measure script, a output png file
- Generate a graph based on benchmark outputs from the measure script

## Kernel List

On AMD :

- rocblas 
- rocblas_wo_dt

On NVIDIA :

- cublas
- cublas_wo_dt 


## Documentation


## Usage/Examples

By using the script :

```bash
./script/measure.sh {options} <kernels>
```

Example :
```bash
./script/measure.sh -p rocblas_wo_dt -vS
```
will run a 1 benchmark of the kernel rocblas_wo_dt in simple precision in verbose and will generate a graph of the result

Use the '-h' option for more information

```bash
./script/check.sh {options} -mXX -kXX <kernels>
```
```bash
./script/check.sh -m5000 rocblas
```
will check rocblas kernel for a 5000x100 matrix and compare it with the cblas kernel

Use the '-h' option for more information

```
.......''',,,',;::ccccc:;,'............ 
........''';cldkO000KK00Oxoc;'..........
''''''..',cxO0000KKKKKK00000kdc'........
,,,,,,,;lk0Ooccd0KKKKKKK0klcokOd:'......
,,,,,,:x00d'   .:OKKKKK0o.   .lOkl'.....
''''';x00x'     .oK0000d.     .lOko'....
.''',o000l       :00000c       ;kkkl....
....:OK00:       :0K000c       ;kkkd,...
 ..'l0000l      .oKKKKKo.      :kkkx, ..
   .l000Kk,    .c0KKKKKO;     'dkkkk;  .
   .:O0000kl;;:d0KK00000kc'.':xkkkkx;...
....;k0000OOOO00000000000Okxxkkkkkkd,...
....'lOOOOOOOOOO0000000OOOOkxxkkxxxc....
.....,dOOkkOOOOOOO0000OOkkkkkxxxxxl.....
......,dkkOOOOOOOOkxdxOOkkkkkxxxdc.... .
........cxkOkkkkkx:..;dxkkkkOkxo,. .....
.........':odxdddolccloooddddo;.     ...  
............,;cllloooolllc:,..  ........        
..................'''.....  ............        
```
