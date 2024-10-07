#!/bin/bash

./run_bgemm_square.sh power 1 32768 2 4 vtrain bf16
./run_bgemm_square.sh power 1 32768 2 8 vtrain bf16
./run_bgemm_square.sh power 1 32768 2 16 vtrain bf16
./run_bgemm_square.sh power 1 32768 2 32 vtrain bf16
./run_bgemm_square.sh power 1 32768 2 64 vtrain bf16
./run_bgemm_square.sh power 1 32768 2 128 vtrain bf16
./run_bgemm_square.sh power 1 32768 2 256 vtrain bf16
./run_bgemm_square.sh power 1 32768 2 512 vtrain bf16
./run_bgemm_square.sh power 1 32768 2 1024 vtrain bf16
./run_bgemm_square.sh power 1 32768 2 2048 vtrain bf16