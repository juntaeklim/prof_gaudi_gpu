#!/bin/bash
fixed_sizes=(1 2 4 8 16 32 64 128 256)

# Run the loop for M, K, N values
for fixed_size in "${fixed_sizes[@]}"
do
    echo "$fixed_size"
    python parser_fixed_n.py --type power --start 256 --end 65536 --stride 2 --method vtrain --dtype bf16 --fixed-size $fixed_size
done
