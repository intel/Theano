#!/bin/bash


source ~/.bashrc
#export KMP_BLOCKTIME=1
#export KMP_AFFINITY=verbose,granularity=core,noduplicates,compact,0,0
#export OMP_NUM_THREADS=66
#export MKL_DYNAMIC=false


./a.out --M=16 --P=16 --N=4 --loop=50000
./a.out --M=16 --P=16 --N=8 --loop=50000
./a.out --M=16 --P=16 --N=12 --loop=50000
./a.out --M=16 --P=16 --N=16 --loop=50000
./a.out --M=16 --P=16 --N=32 --loop=50000
./a.out --M=16 --P=16 --N=64 --loop=50000
./a.out --M=16 --P=16 --N=96 --loop=50000
./a.out --M=16 --P=16 --N=128 --loop=50000
./a.out --M=16 --P=16 --N=256 --loop=50000



