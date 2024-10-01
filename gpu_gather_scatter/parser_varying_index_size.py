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
    parser.add_argument("--fixed-size", type=int, default=4096)
    parser.add_argument("--bench", type=str, choices=["gather_s", "gather_v", "scatter_s", "scatter_v", "gups_gather", "gups_update"])
    parser.add_argument("--dim-size", type=int, default=64)
    parser.add_argument("--custom", action="store_true", default=False)
    args = parser.parse_args()

    pattern = args.type
    start = args.start
    end = args.end
    stride = args.stride
    method = args.method
    fixed_size = args.fixed_size
    bench = args.bench
    bench_list = bench.split("_")
    dim_size = args.dim_size
    custom = args.custom
    
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

    if bench_list[1] == "s" or bench_list[0] == "gups":
        print("Result of (fixed_size) = (%d) for %s" %(fixed_size, method))
    elif bench_list[1] == "v":
        print("Result of (fixed_size, dim) = (%d, %d) for %s" %(fixed_size, dim_size, method))
    else:
        assert False
        
    print()
    if method == "vtrain":
        if bench == "gups_update":
            print("number of inputs, number of indices, kernel_0 (us), interval_0 (us), kernel_1 (us), interval_1 (us), kernel_2 (us), interval_2 (us), kernel_3 (us), interval_3 (us), kernel_4 (us)")
        else:
            print("number of inputs, number of outputs, time_1st_kernel (us), time_interval (us), time_2nd_kernel (us)")
    elif method == "time":
        print("number of inputs, number of outputs, time (us)")
    else:
        assert False
        
    for variable in variables:
        if bench in ["gather_s", "gather_v", "gups_gather", "gups_update"]:
            file_name = "%s/%s_%s_input_%d_output_%d_dim_%d.txt" %(log_path, bench, method, fixed_size, variable, dim_size)
        elif bench in ["scatter_s", "scatter_v"]:
            file_name = "%s/%s_%s_input_%d_output_%d_dim_%d.txt" %(log_path, bench, method, variable, fixed_size, dim_size)
        else:
            assert False

        with open(file_name, "r") as f:
            results = f.readlines()[-1].split(",")
            if method == "vtrain":
                if custom:
                    assert len(results) == 3
                    print("%d, %d, %f" %(int(results[0]), int(results[1]), float(results[2])))
                elif bench == "gups_update":
                    assert len(results) == 11
                    print("%d, %d, %f, %f, %f, %f, %f, %f, %f, %f, %f" %(int(results[0]), int(results[1]), float(results[2]), float(results[3]), float(results[4]), float(results[5]), float(results[6]), float(results[7]), float(results[8]), float(results[9]), float(results[10])))
                else:
                    assert len(results) == 5
                    print("%d, %d, %f, %f, %f" %(int(results[0]), int(results[1]), float(results[2]), float(results[3]), float(results[4])))
            elif method == "time":
                assert len(results) == 3
                print("%d, %d, %f" %(int(results[0]), int(results[1]), float(results[2])))
            else:
                assert False
            

if __name__ == "__main__":
	run()