import torch
import argparse
import time
import vtrain_profiler as vp
from custom_gather import gather_v_i32_indices
from custom_scatter import scatter_v_i32_indices

# @torch.compile
# def mul(a, b):
#     return torch.mm(a, b)

def run():
    parser = argparse.ArgumentParser(description="Gather GPU test")
 
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--gpu-n", type=int, default=0)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--method", type=str, choices=["time", "vtrain", "nsys", "ncomp"])
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32", "tf32"])
    args = parser.parse_args()
    
    M = args.M
    K = args.K
    N = args.N
    gpu_n = args.gpu_n
    test = args.test
    method = args.method
    if args.dtype == "bf16":
        dtype = torch.bfloat16
        max_flops = 312
    elif args.dtype == "fp16":
        dtype = torch.float16
        max_flops = 312
    elif args.dtype == "fp32":
        dtype = torch.float32
        max_flops = 19.5
    elif args.dtype == "tf32":
        max_flops = 156
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        dtype = torch.float32
    else:
        assert False
        
    device = torch.device("cuda:%d" %gpu_n)
    
    if test:
        n_warmup = 1
        n_iter = 0

        M = 4
        K = 4
        N = 4        
    else:
        # Profiler #############################
        if method == "vtrain" or method == "time":
            n_warmup = 15
            n_iter = 5
            
            if method == "vtrain":
                start_times = []
                durations = []  
                kernel_names = []
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

    # mm = torch.compile(torch.mm, fullgraph=True, dynamic=False)
    
    for i in range(n_warmup + n_iter):
        # Tensor preparation ###################
        input_0_tensor_cpu = torch.arange(M * K, dtype=dtype).view(M, K)
        input_1_tensor_cpu = torch.arange(K * N, dtype=dtype).view(K, N)
        
        input_0_tensor = input_0_tensor_cpu.to(device)
        input_1_tensor = input_1_tensor_cpu.to(device)
        ########################################
        
        # Profiler #############################
        if not test and method == "time" and i >= n_warmup:
            torch.cuda.synchronize(device=device)
            start = time.time()
        ########################################
        
        ## Kernel execution ####################
        output_tensor = torch.mm(input_0_tensor, input_1_tensor)
        ########################################
        
        # Profiler #############################
        if not test and method == "time" and i >= n_warmup:
            torch.cuda.synchronize(device=device)
            end = time.time()
            result.append((end - start) * (10**6)) # sec -> usec
        elif not test and method == "vtrain" and i % 10 == 9:
            torch.cuda.synchronize(device=device)
            tmp = vp.finish_trace()
            if tmp:
                trace = tmp.strip().split('\n')
                for j in range(len(trace)):
                    row = trace[j]
                    splitted_row = row.split(",")
                    if splitted_row[2] == "KERNEL":
                        start_times.append(float(splitted_row[0]) / 1000)
                        durations.append(float(splitted_row[1]) / 1000)
                        kernel_names.append(splitted_row[3])
            vp.init_trace()
        ########################################
        
        if test:
            print()
            print("input_0_tensor")
            print(input_0_tensor)
            print(input_0_tensor.dtype)
            print()
            print("input_1_tensor")
            print(input_1_tensor)
            print(input_1_tensor.dtype)
            print()
            
            print("output_tensor")
            print(output_tensor)
            print(output_tensor.dtype)
            
    # Profiler #############################
    if not test and method == "time":
        assert len(result) == n_iter
        print("result")
        print(result)
        result_tensor = torch.tensor(result)
        
        final_time = result_tensor.mean()
        tflops = 2*M*K*N/final_time/(10**6)
        print("Performance: %f FLOPS, %f %%" %(tflops, tflops/max_flops*100))
        print("M, K, N, time (us)")
        print("%d, %d, %d, %f" %(M, K, N, final_time))
    elif not test and method == "vtrain":
        torch.cuda.synchronize(device=device)
        trace = vp.finish_trace().strip().split('\n')
        for i in range(len(trace)):
            row = trace[i]
            splitted_row = row.split(",")
            if splitted_row[2] == "KERNEL":
                start_times.append(float(splitted_row[0]) / 1000)
                durations.append(float(splitted_row[1]) / 1000)
        
        if len(start_times) == (n_warmup + n_iter):
            result_0 = []
            
            for i in range(n_warmup + n_iter):
                result_0.append(durations[i])
            
            print("result_0")
            print(result_0)

            result_0_tensor = torch.tensor(result_0)
            
            final_time_0 = result_0_tensor[-n_iter:].mean()

            print()
            tflops = 2*M*K*N/final_time_0/(10**6)
            print("Performance: %f FLOPS, %f %%" %(tflops, tflops/max_flops*100))
            print("M, K, N, kernel_duration (us)")
            print("%d, %d, %d, %f" %(M, K, N, final_time_0)) 
        elif len(start_times) == 2 * (n_warmup + n_iter):
            result_0 = []
            result_1 = []
            result_2 = []
            
            for i in range(n_warmup + n_iter):
                result_0.append(durations[2*i+0])
                result_1.append(start_times[2*i+1] - start_times[2*i+0] - durations[2*i+0])
                result_2.append(durations[2*i+1])
            
            assert len(result_0) == len(result_1)
            assert len(result_1) == len(result_2)
            
            result_0_tensor = torch.tensor(result_0)
            result_1_tensor = torch.tensor(result_1)
            result_2_tensor = torch.tensor(result_2)
            
            final_time_0 = result_0_tensor[-n_iter:].mean()
            final_time_1 = result_1_tensor[-n_iter:].mean()
            final_time_2 = result_2_tensor[-n_iter:].mean()
            
            print()
            tflops = 2*M*K*N/(final_time_0 + final_time_1 + final_time_2)/(10**6)
            print("Performance: %f FLOPS, %f %%" %(tflops, tflops/max_flops*100))
            print("M, K, N, kernel_duration_0 (us), interval (us), kernel_duration_1 (us)")
            print("%d, %d, %d, %f, %f, %f" %(M, K, N, final_time_0, final_time_1, final_time_2)) 
        else:
            assert False
    elif not test and method == "nsys":
        torch.cuda.cudart().cudaProfilerStop()
    ########################################
    
if __name__ == "__main__":
    run()