#!/bin/bash

/opt/nvidia/nsight-systems/2024.6.1/bin/nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --gpu-metrics-device=all --cudabacktrace=true --osrt-threshold=10000 --cuda-memory-usage=true --capture-range=cudaProfilerApi -o ~/prof_gaudi_gpu/gpu_llm/out -f true -x true python run_trt_multiple_configs.py --model meta-llama/Llama-3.1-8B-Instruct --n-gpus 1 

