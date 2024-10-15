#!/bin/bash

./run_gather_scatter.sh power 16384  2097152 2 4194304 vtrain gather_v 1
./run_gather_scatter.sh power 16384  2097152 2 4194304 vtrain gather_v 2
./run_gather_scatter.sh power 16384  2097152 2 4194304 vtrain gather_v 4
./run_gather_scatter.sh power 16384  2097152 2 4194304 vtrain gather_v 8
./run_gather_scatter.sh power 16384  2097152 2 4194304 vtrain gather_v 16
./run_gather_scatter.sh power 16384  2097152 2 4194304 vtrain gather_v 32
./run_gather_scatter.sh power 16384  2097152 2 4194304 vtrain gather_v 64
./run_gather_scatter.sh power 16384  2097152 2 4194304 vtrain gather_v 128
./run_gather_scatter.sh power 16384  2097152 2 4194304 vtrain gather_v 256
./run_gather_scatter.sh power 16384  2097152 2 4194304 vtrain gather_v 512