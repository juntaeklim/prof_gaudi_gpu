#!/bin/bash
dtype=$1
fixed_sizes=(8 16 32 64 128 256)

python parser_square.py --type power --start 32 --end 32768 --stride 2 --method vtrain --dtype $dtype
# Run the loop for M, K, N values
for fixed_size in "${fixed_sizes[@]}"
do
    python parser_fixed_n.py --type power --start 256 --end 32768 --stride 2 --method vtrain --dtype $dtype --fixed-size $fixed_size
done
