import torch
import argparse
import time
import vtrain_profiler as vp
from custom_gather import gather_v_i32_indices
from custom_scatter import scatter_v_i32_indices

def run():
    parser = argparse.ArgumentParser(description="Gather GPU test")
 
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=4)
    parser.add_argument("--dim-size", type=int, default=64)
    parser.add_argument("--gpu-n", type=int, default=0)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--method", type=str, choices=["time", "vtrain", "nsys", "ncomp"])
    parser.add_argument("--bench", type=str, choices=["gather_s", "gather_v", "scatter_s", "scatter_v", "gups_gather", "gups_update"])
    parser.add_argument("--using-prepared-indices", action="store_true", default=False)
    parser.add_argument("--algo", type=str, choices=["randint"])
    parser.add_argument("--custom", action="store_true", default=False)
    args = parser.parse_args()
    
    input_size = args.input_size
    output_size = args.output_size
    dim_size = args.dim_size
    gpu_n = args.gpu_n
    test = args.test
    method = args.method
    bench = args.bench
    using_prepared_indices = args.using_prepared_indices
    algo = args.algo
    custom = args.custom
    
    device = torch.device("cuda:%d" %gpu_n)
    bench_list = bench.split("_")
    
    if test:
        n_warmup = 1
        n_iter = 0
        
        if bench_list[0] == "gather" or bench == "gups_gather":
            input_size = 6
            output_size = 3
            if bench_list[1] == "v":
                dim_size = 2
        elif bench_list[0] == "scatter" or bench == "gups_update":
            input_size = 3
            output_size = 6
            if bench_list[1] == "v":
                dim_size = 2
    else:
        # Profiler #############################
        if method == "vtrain" or method == "time":
            n_warmup = 15
            n_iter = 5
            
            if method == "vtrain":
                vp.init_trace()
            else:
                result = []
        elif method == "nsys" or method =="ncomp":
            n_warmup = 2
            n_iter = 0
            
            if method == "nsys":
                torch.cuda.cudart().cudaProfilerStart()
        else:
            assert False
        ########################################

    
    for i in range(n_warmup + n_iter):
        # Tensor preparation ###################
        if bench_list[0] == "gather":
            assert input_size >= output_size
            if not using_prepared_indices:
                index_tensor_cpu = torch.randint(low=0, high=input_size, size=(output_size,), dtype=torch.int32)
            if bench_list[1] == "s":
                input_tensor_cpu = torch.arange(input_size, dtype=torch.float)
            elif bench_list[1] == "v":
                input_tensor_cpu = torch.arange(input_size * dim_size, dtype=torch.float).view(input_size, dim_size)
            else:
                assert False
        elif bench_list[0] == "scatter":
            assert input_size <= output_size
            if not using_prepared_indices:
                index_tensor_cpu = torch.randint(low=0, high=output_size, size=(input_size,), dtype=torch.int32)
            if bench_list[1] == "s":
                input_tensor_cpu = torch.arange(input_size, dtype=torch.float)
                output_tensor_cpu = torch.zeros(output_size, dtype=input_tensor_cpu.dtype)
            elif bench_list[1] == "v":
                input_tensor_cpu = torch.arange(input_size * dim_size, dtype=torch.float).view(input_size, dim_size)
                output_tensor_cpu = torch.zeros(output_size, dim_size, dtype=input_tensor_cpu.dtype)
            else:
                assert False
        elif bench_list[0] == "gups":
            A = 1664525  
            C = 12345
            M = 2**32

            if not using_prepared_indices:
                index_tensor_cpu = torch.zeros(output_size, dtype=torch.int32)
                seed = torch.randint(0, M, (1,), dtype=torch.uint32).item()
                for i in range(output_size):
                    seed = (A * seed + C) % M
                    index_tensor_cpu[i] = seed % input_size
            if bench_list[1] == "gather":
                input_tensor_cpu = torch.arange(input_size, dtype=torch.float32)
            elif bench_list[1] == "update":
                input_tensor_cpu = torch.arange(input_size, dtype=torch.int32)
            else:
                assert False
        else:
            assert False
        
        if using_prepared_indices:
            if bench in ["gather_s", "gather_v", "gups_gather", "gups_update"]:
                index_tensor_cpu = torch.load("./index_tensors/%s_entire_%d_index_%d_iter_%d" %(algo, input_size, output_size, i), weights_only=True)
            elif bench in ["scatter_s", "scatter_v"]:
                index_tensor_cpu = torch.load("./index_tensors/%s_entire_%d_index_%d_iter_%d" %(algo, output_size, input_size, i), weights_only=True)
            else:
                assert False
            assert index_tensor_cpu.device == torch.device("cpu")

        input_tensor = input_tensor_cpu.to(device)
        index_tensor = index_tensor_cpu.to(device)
        if bench_list[0] == "scatter":
            output_tensor = output_tensor_cpu.to(device)
        ########################################
        
        # Profiler #############################
        if not test and method == "time" and i >= n_warmup:
            torch.cuda.synchronize(device=device)
            start = time.time()
        ########################################
        
        ## Kernel execution ####################
        if bench_list[0] == "gather":
            if custom:
                output_tensor = gather_v_i32_indices(input_tensor, index_tensor)
                print(output_tensor)
            else:
                output_tensor = input_tensor[index_tensor]
        elif bench_list[0] == "scatter":
            if custom:
                output_tensor = scatter_v_i32_indices(input_tensor, index_tensor, output_tensor)
            else:
                output_tensor[index_tensor] = input_tensor
        elif bench_list[0] == "gups":
            if bench_list[1] == "gather":
                output_tensor = input_tensor[index_tensor]
            elif bench_list[1] == "update":
                input_tensor[index_tensor] = torch.bitwise_xor(input_tensor[index_tensor], index_tensor)
            else:
                assert False
        else:
            assert False
        ########################################
        
        # Profiler #############################
        if not test and method == "time" and i >= n_warmup:
            torch.cuda.synchronize(device=device)
            end = time.time()
            result.append((end - start) * (10**6)) # sec -> usec
        ########################################
        
        if test:
            print()
            print("input_tensor")
            print(input_tensor)
            print(input_tensor.dtype)
            print()
            print("index_tensor")
            print(index_tensor)
            print(index_tensor.dtype)
            print()
            
            if bench in ["gather_s", "gather_v", "scatter_s", "scatter_v", "gups_gather"]:
                print("output_tensor")
                print(output_tensor)
                print(output_tensor.dtype)
        elif bench == "gups_update":
            input_tensor_cpu_1 = input_tensor.to("cpu")
            print(input_tensor_cpu_1.device)
        else:
            output_tensor_cpu = output_tensor.to("cpu")
            print(output_tensor_cpu.device)
    
    # Profiler #############################
    if not test and method == "time":
        assert len(result) == n_iter
        print("result")
        print(result)
        result_tensor = torch.tensor(result)
        
        print("number of inputs, number of outputs, time (us)")
        final_time = result_tensor.mean()
        print("%d, %d, %f" %(input_size, output_size, final_time))
    elif not test and method == "vtrain":
        trace = vp.finish_trace().strip().split('\n')
        start_times = []
        durations = []
        for i in range(len(trace)):
            row = trace[i]
            splitted_row = row.split(",")
            if splitted_row[2] == "KERNEL":
                start_times.append(float(splitted_row[0]) / 1000)
                durations.append(float(splitted_row[1]) / 1000)
        
        if custom:
            assert len(start_times) == n_warmup + n_iter
            assert len(durations) == n_warmup + n_iter
        elif bench == "gups_update":
            assert len(start_times) == 5 * (n_warmup + n_iter)
            assert len(durations) == 5 * (n_warmup + n_iter)
        else:
            assert len(start_times) == 2 * (n_warmup + n_iter)
            assert len(durations) == 2 * (n_warmup + n_iter)
        
        result_0 = []
        
        if not custom:
            result_1 = []
            result_2 = []
            
        if not custom and bench == "gups_update":
            result_3 = []
            result_4 = []
            result_5 = []
            result_6 = []
            result_7 = []
            result_8 = []
            
        for i in range(n_warmup + n_iter):
            result_0.append(durations[i])
            
            if not custom and bench != "gups_update":
                result_0.append(durations[5*i])
                result_1.append(start_times[5*i + 1] - start_times[5*i] - durations[5*i])
                result_2.append(durations[5*i + 1])
                result_3.append(start_times[5*i + 2] - start_times[5*i + 1] - durations[5*i + 1])
                result_4.append(durations[5*i + 2])
                result_5.append(start_times[5*i + 3] - start_times[5*i + 2] - durations[5*i + 2])
                result_6.append(durations[5*i + 3])
                result_7.append(start_times[5*i + 4] - start_times[5*i + 3] - durations[5*i + 3])
                result_8.append(durations[5*i + 4])
                
            if not custom and bench == "gups_update":
                result_0.append(durations[2*i])
                result_1.append(start_times[2*i + 1] - start_times[2*i] - durations[2*i])
                result_2.append(durations[2*i + 1])
        
        print("result_0")
        print(result_0)
        if not custom:
            print("result_1")
            print(result_1)
            print("result_2")
            print(result_2)
            
        if not custom and bench == "gups_update":
            print("result_3")
            print(result_3)
            print("result_4")
            print(result_4)
            print("result_5")
            print(result_5)
            print("result_6")
            print(result_6)
            print("result_7")
            print(result_7)
            print("result_8")
            print(result_8)
        
        result_0_tensor = torch.tensor(result_0)
        
        if not custom:
            result_1_tensor = torch.tensor(result_1)
            result_2_tensor = torch.tensor(result_2)
        
        if not custom and bench == "gups_update":
            result_3_tensor = torch.tensor(result_3)
            result_4_tensor = torch.tensor(result_4)
            result_5_tensor = torch.tensor(result_5)
            result_6_tensor = torch.tensor(result_6)
            result_7_tensor = torch.tensor(result_7)
            result_8_tensor = torch.tensor(result_8)
            
        final_time_0 = result_0_tensor[-n_iter:].mean()

        if not custom:
            final_time_1 = result_1_tensor[-n_iter:].mean()
            final_time_2 = result_2_tensor[-n_iter:].mean()
        
        if not custom and  bench == "gups_update":
            final_time_3 = result_3_tensor[-n_iter:].mean()
            final_time_4 = result_4_tensor[-n_iter:].mean()
            final_time_5 = result_5_tensor[-n_iter:].mean()
            final_time_6 = result_6_tensor[-n_iter:].mean()
            final_time_7 = result_7_tensor[-n_iter:].mean()
            final_time_8 = result_8_tensor[-n_iter:].mean()
        print()
        if custom:
            print("number of inputs, number of indices, kernel_duration (us)")
            print("%d, %d, %f" %(input_size, output_size, final_time_0)) 
        elif bench == "gups_update":
            print("number of inputs, number of indices, kernel_0 (us), interval_0 (us), kernel_1 (us), interval_1 (us), kernel_2 (us), interval_2 (us), kernel_3 (us), interval_3 (us), kernel_4 (us), interval_4 (us), kernel_5 (us)")
            print("%d, %d, %f, %f, %f, %f, %f, %f, %f, %f, %f" %(input_size, output_size, final_time_0, final_time_1, final_time_2, final_time_3, final_time_4, final_time_5, final_time_6, final_time_7, final_time_8))
        else:
            print("number of inputs, number of outputs, time_1st_kernel (us), time_interval (us), time_2nd_kernel (us)")
            print("%d, %d, %f, %f, %f" %(input_size, output_size, final_time_0, final_time_1, final_time_2))
    elif not test and method == "nsys":
        torch.cuda.cudart().cudaProfilerStop()
    ########################################
    
if __name__ == "__main__":
    run()