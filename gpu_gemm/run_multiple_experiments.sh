#!/bin/bash

./run_gemm_fix_n.sh power 65536 65536 2 vtrain bf16 1
./run_gemm_fix_n.sh power 65536 65536 2 vtrain bf16 2
./run_gemm_fix_n.sh power 65536 65536 2 vtrain bf16 4
./run_gemm_fix_n.sh power 65536 65536 2 vtrain bf16 8
./run_gemm_fix_n.sh power 65536 65536 2 vtrain bf16 16
./run_gemm_fix_n.sh power 65536 65536 2 vtrain bf16 32
./run_gemm_fix_n.sh power 65536 65536 2 vtrain bf16 64
./run_gemm_fix_n.sh power 65536 65536 2 vtrain bf16 128
./run_gemm_fix_n.sh power 65536 65536 2 vtrain bf16 256

