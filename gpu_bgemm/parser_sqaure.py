import os
import argparse
import pandas as pd
import re

def run():
    parser = argparse.ArgumentParser("Extract the data from log files")
    parser.add_argument("--type", type=str, choices=["linear", "power"])
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--method", type=str, choices=["time", "vtrain", "nsys", "ncomp"])
    parser.add_argument("--dtype", type=str, choices=["fp16", "fp32", "bf16", "tf32"])
    args = parser.parse_args()

    pattern = args.type
    start = args.start
    end = args.end
    stride = args.stride
    M = args.M
    K = M
    N = M
    # K = args.K
    # N = args.N
    method = args.method
    dtype = args.dtype
    
    if pattern == "linear":
        assert (end - start) % stride == 0
        values = [start + stride * i for i in range(int((end - start)/stride) + 1)]
    elif pattern == "power":
        values = []
        tmp = start
        while tmp <= end:
            values.append(tmp)
            tmp = tmp * stride
        assert tmp / stride == end

    log_path = "./logs"

    try:
        file_list = os.listdir(log_path)
    except FileNotFoundError:
        print(f"The directory {log_path} was not found.")
        return

    variables = values

    print("Result of square GEMM for %s" %(method))
    print()
    if method == "vtrain":
        print("B, M, K, N, time (us)")
    elif method == "time":
        print("B, M, K, N, time (us)")
    else:
        assert False
        
    for variable in variables:
        file_name = "%s/bgemm_%s_b_%d_m_%d_k_%d_n_%d_dtype_%s.txt" %(log_path, method, variable, M, K, N, dtype)

        with open(file_name, "r") as f:
            results = f.readlines()[-1].split(",")
            if method == "vtrain":
                assert len(results) == 5
                print("%d, %d, %d, %d, %f" %(int(results[0]), int(results[1]), int(results[2]), int(results[3]), float(results[4])))
            elif method == "time":
                assert len(results) == 5
                print("%d, %d, %d, %d, %f" %(int(results[0]), int(results[1]), int(results[2]), int(results[3]), float(results[4])))
            else:
                assert False
            

if __name__ == "__main__":
	run()