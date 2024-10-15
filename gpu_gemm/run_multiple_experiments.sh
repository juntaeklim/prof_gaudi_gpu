#!/bin/bash

./run_gemm_fix_n.sh power 256 32768 2 vtrain fp16 1
./run_gemm_fix_n.sh power 256 32768 2 vtrain fp16 2
./run_gemm_fix_n.sh power 256 32768 2 vtrain fp16 4
./run_gemm_fix_n.sh power 256 32768 2 vtrain fp16 8
./run_gemm_fix_n.sh power 256 32768 2 vtrain fp16 16
./run_gemm_fix_n.sh power 256 32768 2 vtrain fp16 32
./run_gemm_fix_n.sh power 256 32768 2 vtrain fp16 64
./run_gemm_fix_n.sh power 256 32768 2 vtrain fp16 128
./run_gemm_fix_n.sh power 256 32768 2 vtrain fp16 256

