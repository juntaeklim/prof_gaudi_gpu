import torch
import argparse
import time
import vtrain_profiler as vp

def run():
    parser = argparse.ArgumentParser(description="Gather GPU test")
 
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=4)
    parser.add_argument("--gpu-n", type=int, default=0)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--method", type=str, choices=["time", "vtrain", "nsys", "ncomp"])
    parser.add_argument("--bench", type=str, choices=["gather_s", "gather_v", "scatter_s", "scatter_v", "gups"])
    args = parser.parse_args()
    
    input_size = args.input_size
    output_size = args.output_size
    gpu_n = args.gpu_n
    test = args.test
    method = args.method
    bench = args.bench
    dim_size = 64
    
    device = torch.device("cuda:%d" %gpu_n)
    
    if test:
        bench_list = bench.split("_")
        
        if bench_list[0] == "gather":
            input_size = 6
            output_size = 3
            
            if bench_list[1] == "s":
                input_tensor = torch.arange(input_size, device=device, dtype=torch.float)
                index_tensor = torch.randint(low=0, high=input_size, size=(output_size,), device=device, dtype=torch.int32)
            elif bench_list[1] == "v":
                dim_size = 2
                input_tensor = torch.arange(input_size * dim_size, device=device, dtype=torch.float).view(input_size, dim_size)
                index_tensor = torch.randint(low=0, high=input_size, size=(output_size,), device=device, dtype=torch.int32)
                
            output_tensor = input_tensor[index_tensor]
            
        elif bench_list[0] == "scatter":
            input_size = 3
            output_size = 6
            
            if bench_list[1] == "s":
                input_tensor = torch.arange(input_size, device=device, dtype=torch.float)
                index_tensor = torch.randint(low=0, high=output_size, size=(input_size,), device=device, dtype=torch.int32)
                output_tensor = torch.zeros(output_size, device=device, dtype=input_tensor.dtype)
            elif bench_list[1] == "v":
                dim_size = 2
                input_tensor = torch.arange(input_size * dim_size, device=device, dtype=torch.float).view(input_size, dim_size)
                index_tensor = torch.randint(low=0, high=output_size, size=(input_size,), device=device, dtype=torch.int32)
                output_tensor = torch.zeros(output_size, dim_size, device=device, dtype=input_tensor.dtype)
                output_tensor[index_tensor] = input_tensor
            
            output_tensor[index_tensor] = input_tensor
        elif bench_list[0] == "gups":
            input_size = 6
            output_size = 3
            
            A = 1664525  
            C = 12345
            M = 2**32

            index_tensor = torch.zeros(output_size, dtype=torch.int32, device=device)
            seed = torch.randint(0, M, (1,), dtype=torch.uint32).item()
            for i in range(output_size):
                seed = (A * seed + C) % M
                index_tensor[i] = seed % input_size
                
            input_tensor = torch.arange(input_size, dtype=torch.float, device=device)
            
            output_tensor = input_tensor[index_tensor]
        else:
            assert False
            
        print("input_tensor")
        print(input_tensor)
        print(input_tensor.dtype)
        print()
        print("index_tensor")
        print(index_tensor)
        print(index_tensor.dtype)
        print()
        print("output_tensor")
        print(output_tensor)
        print(output_tensor.dtype)
        print()
    else:
        # Profiler #############################
        if method == "vtrain" or method == "time":
            n_warmup = 3
            n_iter = 4
            
            if method == "vtrain":
                vp.init_trace()
            else:
                result = []
        elif method == "nsys" or method =="ncomp":
            n_warmup = 3
            n_iter = 0
            
            if method == "nsys":
                torch.cuda.cudart().cudaProfilerStart()
        else:
            assert False
        ########################################

        bench_list = bench.split("_")
        
        for i in range(n_warmup + n_iter):
            # Tensor preparation ###################
            if bench_list[0] == "gather":
                assert input_size >= output_size
                index_tensor_cpu = torch.randint(low=0, high=input_size, size=(output_size,), dtype=torch.int32)
                if bench_list[1] == "s":
                    input_tensor_cpu = torch.arange(input_size, dtype=torch.float)
                elif bench_list[1] == "v":
                    input_tensor_cpu = torch.arange(input_size * dim_size, dtype=torch.float).view(input_size, dim_size)
                else:
                    assert False
            elif bench_list[0] == "scatter":
                assert input_size <= output_size
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

                index_tensor_cpu = torch.zeros(output_size, dtype=torch.int32)
                seed = torch.randint(0, M, (1,), dtype=torch.uint32).item()
                for i in range(output_size):
                    seed = (A * seed + C) % M
                    index_tensor_cpu[i] = seed % input_size
                    
                input_tensor_cpu = torch.arange(input_size, dtype=torch.float)
            else:
                assert False
                
            input_tensor = input_tensor_cpu.to(device)
            index_tensor = index_tensor_cpu.to(device)
            if bench_list[0] == "scatter":
                output_tensor = output_tensor_cpu.to(device)
            ########################################
            
            # Profiler #############################
            if method == "time" and i >= n_warmup:
                torch.cuda.synchronize(device=device)
                start = time.time()
            ########################################
            
            ## Kernel execution ####################
            if bench_list[0] == "gather" or bench_list[0] == "gups":
                output_tensor = input_tensor[index_tensor]
            elif bench_list[0] == "scatter":
                output_tensor[index_tensor] = input_tensor
            else:
                assert False
            ########################################
            
            # Profiler #############################
            if method == "time" and i >= n_warmup:
                torch.cuda.synchronize(device=device)
                end = time.time()
                result.append((end - start) * (10**6)) # sec -> usec
            ########################################
            
            output_tensor_cpu = output_tensor.to("cpu")
            print(output_tensor_cpu.device)
            
        # Profiler #############################
        if method == "time":
            assert len(result) == n_iter
            print("result")
            print(result)
            result_tensor = torch.tensor(result)
            
            print("number of inputs, number of outputs, time (us)")
            final_time = result_tensor.mean()
            print("%d, %d, %f" %(input_size, output_size, final_time))
        elif method == "vtrain":
            trace = vp.finish_trace().strip().split('\n')
            start_times = []
            durations = []
            for i in range(len(trace)):
                row = trace[i]
                splitted_row = row.split(",")
                if splitted_row[2] == "KERNEL":
                    start_times.append(float(splitted_row[0]) / 1000)
                    durations.append(float(splitted_row[1]) / 1000)
                    
            assert len(start_times) == 2 * (n_warmup + n_iter)
            assert len(durations) == 2 * (n_warmup + n_iter)
            
            result_0 = []            
            result_1 = []
            result_2 = []
            
            for i in range(n_warmup + n_iter):
                result_0.append(durations[2*i])
                result_1.append(start_times[2*i + 1] - start_times[2*i] - durations[2*i])
                result_2.append(durations[2*i + 1])
            
            print("result_0")
            print(result_0)
            print("result_1")
            print(result_1)
            print("result_2")
            print(result_2)
            
            result_0_tensor = torch.tensor(result_0)
            result_1_tensor = torch.tensor(result_1)
            result_2_tensor = torch.tensor(result_2)
            
            final_time_0 = result_0_tensor[-n_iter:].mean()
            final_time_1 = result_1_tensor[-n_iter:].mean()
            final_time_2 = result_2_tensor[-n_iter:].mean()
            
            print()
            print("number of inputs, number of outputs, time_1st_kernel (us), time_interval (us), time_2nd_kernel (us)")
            print("%d, %d, %f, %f, %f" %(input_size, output_size, final_time_0, final_time_1, final_time_2))
        elif method == "nsys":
            torch.cuda.cudart().cudaProfilerStop()
        ########################################
if __name__ == "__main__":
    run()