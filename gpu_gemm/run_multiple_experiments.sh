#!/bin/bash

./run_gemm_square.sh power 32 32768 2 vtrain fp32
./run_gemm_fix_n.sh power 256 65536 2 vtrain fp32 1
./run_gemm_fix_n.sh power 256 65536 2 vtrain fp32 2
./run_gemm_fix_n.sh power 256 65536 2 vtrain fp32 4
./run_gemm_fix_n.sh power 256 65536 2 vtrain fp32 8
./run_gemm_fix_n.sh power 256 65536 2 vtrain fp32 16
./run_gemm_fix_n.sh power 256 65536 2 vtrain fp32 32
./run_gemm_fix_n.sh power 256 65536 2 vtrain fp32 64
./run_gemm_fix_n.sh power 256 65536 2 vtrain fp32 128
./run_gemm_fix_n.sh power 256 65536 2 vtrain fp32 256

./run_gemm_square.sh power 32 32768 2 vtrain tf32
./run_gemm_fix_n.sh power 256 65536 2 vtrain tf32 1
./run_gemm_fix_n.sh power 256 65536 2 vtrain tf32 2
./run_gemm_fix_n.sh power 256 65536 2 vtrain tf32 4
./run_gemm_fix_n.sh power 256 65536 2 vtrain tf32 8
./run_gemm_fix_n.sh power 256 65536 2 vtrain tf32 16
./run_gemm_fix_n.sh power 256 65536 2 vtrain tf32 32
./run_gemm_fix_n.sh power 256 65536 2 vtrain tf32 64
./run_gemm_fix_n.sh power 256 65536 2 vtrain tf32 128
./run_gemm_fix_n.sh power 256 65536 2 vtrain tf32 256

./run_gemm_square.sh power 32 32768 2 vtrain bf16
./run_gemm_fix_n.sh power 256 65536 2 vtrain bf16 1
./run_gemm_fix_n.sh power 256 65536 2 vtrain bf16 2
./run_gemm_fix_n.sh power 256 65536 2 vtrain bf16 4
./run_gemm_fix_n.sh power 256 65536 2 vtrain bf16 8
./run_gemm_fix_n.sh power 256 65536 2 vtrain bf16 16
./run_gemm_fix_n.sh power 256 65536 2 vtrain bf16 32
./run_gemm_fix_n.sh power 256 65536 2 vtrain bf16 64
./run_gemm_fix_n.sh power 256 65536 2 vtrain bf16 128
./run_gemm_fix_n.sh power 256 65536 2 vtrain bf16 256
