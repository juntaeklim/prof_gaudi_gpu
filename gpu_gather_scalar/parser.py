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
    parser.add_argument("--method", type=str, default="vtrain")
    parser.add_argument("--input-size", type=int, default=4096)
    args = parser.parse_args()

    pattern = args.type
    start = args.start
    end = args.end
    stride = args.stride
    method = args.method
    input_size = args.input_size
    
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

    index_sizes = values

    print("Result of (input_size) = (%d) for %s" %(input_size, method))
    print()
    for index_size in index_sizes:
        file_name = "%s/%s_input_%d_index_%d.txt" %(log_path, method, input_size, index_size)

        with open(file_name, "r") as f:
            results = f.readlines()[1].split(",")
            print("%d, %f" %(int(results[1]), float(results[-1])))
            

if __name__ == "__main__":
	run()