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
    parser.add_argument("--method", type=str, choices=["time", "vtrain", "nsys", "ncomp"])
    parser.add_argument("--dtype", type=str, choices=["fp16", "fp32", "bf16", "tf32"])
    parser.add_argument("--fixed-size", type=int, default=1)
    
    args = parser.parse_args()

    pattern = args.type
    start = args.start
    end = args.end
    stride = args.stride
    method = args.method
    dtype = args.dtype
    fixed_size = args.fixed_size
    
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

    log_path = "./logs_from_naver"

    try:
        file_list = os.listdir(log_path)
    except FileNotFoundError:
        print(f"The directory {log_path} was not found.")
        return

    variables = values

    # print("Result of GEMV-style GEMM for %s" %(method))
    print()
    if method == "vtrain":
        print("M, K, N, time (us)")
    elif method == "time":
        print("M, K, N, time (us)")
    else:
        assert False
        
    for variable in variables:
        file_name = "%s/gemm_%s_m_%d_k_%d_n_%d_dtype_%s.txt" %(log_path, method, variable, variable, fixed_size, dtype)

        with open(file_name, "r") as f:
            try:
                results = f.readlines()[-1].split(",")
                if method == "vtrain":
                    if len(results) == 4:
                        print("%d, %d, %d, %f" %(int(results[0]), int(results[1]), int(results[2]), float(results[3])))
                    elif len(results) == 6:
                        kernel_0 = float(results[3])
                        interval = float(results[4])
                        kernel_1 = float(results[5])
                        
                        print("%d, %d, %d, %f, %f, %f, %f" %(int(results[0]), int(results[1]), int(results[2]), kernel_0 + interval + kernel_1, kernel_0, interval, kernel_1))
                        
                elif method == "time":
                    assert len(results) == 4
                    print("%d, %d, %d, %f" %(int(results[0]), int(results[1]), int(results[2]), float(results[3])))
                else:
                    assert False
            except:
                continue

if __name__ == "__main__":
	run()