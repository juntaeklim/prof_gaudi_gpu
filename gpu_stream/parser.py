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
    parser.add_argument("--bench", type=str, choices=["copy", "scale", "add", "triad"])
    
    args = parser.parse_args()

    pattern = args.type
    start = args.start
    end = args.end
    stride = args.stride
    method = args.method
    bench = args.bench
    
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

    print("Result for %s with %s" %(bench, method))
    print()
    print("number of inputs, time (us)")
        
    for variable in variables:
        file_name = "%s/%s_%s_input_%d.txt" %(log_path, bench, method, variable)

        with open(file_name, "r") as f:
            results = f.readlines()[-1].strip().split(",")
            assert len(results) == 2
            print("%d, %f" %(int(results[0]), float(results[1])))
            

if __name__ == "__main__":
	run()