import torch
import argparse
import time
import vtrain_profiler as vp

def run():
    parser = argparse.ArgumentParser(description="Gather GPU test")
 
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--gpu-n", type=int, default=0)
    parser.add_argument("--method", type=str, choices=["time", "vtrain", "nsys", "ncomp"])
    parser.add_argument("--bench", type=str, choices=["copy", "scale", "add", "triad"])
    args = parser.parse_args()
    
    input_size = args.input_size
    gpu_n = args.gpu_n
    method = args.method
    bench = args.bench
    
    device = torch.device("cuda:%d" %gpu_n)
    
    # Profiler #############################
    if method == "vtrain" or method == "time":
        n_warmup = 20
        n_iter = 10
        
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

    for i in range(n_warmup + n_iter):
        # Tensor preparation ###################
        input_0_cpu = torch.arange(input_size, dtype=torch.float32)
        input_0 = input_0_cpu.to(device)
        
        if bench in ["add", "triad"]:
            input_1_cpu = torch.arange(input_size, dtype=torch.float32)
            input_1 = input_1_cpu.to(device)
        ########################################
        
        # Profiler #############################
        if method == "time" and i >= n_warmup:
            torch.cuda.synchronize(device=device)
            start = time.time()
        ########################################
        
        ## Kernel execution ####################
        if bench =="copy":
            output = input_0.clone()
        elif bench == "scale":
            output = input_0 * 1.3 # any scalar
        elif bench == "add":
            output = input_0 + input_1
        elif bench == "triad":
            output = torch.add(input_0, input_1, alpha=1.3) # any scalar
        else:
            assert False
        ########################################
        
        # Profiler #############################
        if method == "time" and i >= n_warmup:
            torch.cuda.synchronize(device=device)
            end = time.time()
            result.append((end - start) * (10**6)) # sec -> usec
        ########################################
        
        output_cpu = output.to("cpu")
        print(output_cpu.device)
        
    # Profiler #############################
    if method == "time":
        assert len(result) == n_iter
        print("result")
        print(result)
        result_tensor = torch.tensor(result)
        
        print("number of inputs, time (us)")
        final_time = result_tensor.mean()
        print("%d, %f" %(input_size, final_time))
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
                
        assert len(start_times) == (n_warmup + n_iter)
        assert len(durations) == (n_warmup + n_iter)
        
        result_0 = []            
        
        for i in range(n_warmup + n_iter):
            result_0.append(durations[i])
        
        print("result_0")
        print(result_0)
        
        result_0_tensor = torch.tensor(result_0)
        final_time_0 = result_0_tensor[-n_iter:].mean()
        
        print()
        print("number of inputs, time (us)")
        print("%d, %f" %(input_size, final_time_0))
    elif method == "nsys":
        torch.cuda.cudart().cudaProfilerStop()
    ########################################
if __name__ == "__main__":
    run()