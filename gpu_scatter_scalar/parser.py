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
    if method == "vtrain":
        print("number of inputs, number of indices, time_1st_kernel (us), time_interval (us), time_2nd_kernel (us)")
    elif method == "time":
        print("number of inputs, number of indices, time (us)")
    else:
        assert False
        
    for index_size in index_sizes:
        file_name = "%s/%s_input_%d_index_%d.txt" %(log_path, method, input_size, index_size)

        with open(file_name, "r") as f:
            results = f.readlines()[-1].split(",")
            if method == "vtrain":
                assert len(results) == 5
                print("%d, %d, %f, %f, %f" %(int(results[0]), int(results[1]), float(results[2]), float(results[3]), float(results[4])))
            elif method == "time":
                assert len(results) == 3
                print("%d, %d, %f" %(int(results[0]), int(results[1]), float(results[2])))
            else:
                assert False
            

if __name__ == "__main__":
	run()