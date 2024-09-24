import torch
import argparse
import time
import vtrain_profiler as vp

def run():
    parser = argparse.ArgumentParser(description="Gather GPU test")
 
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--index-size", type=int, default=4)
    parser.add_argument("--dim-size", type=int, default=64)
    parser.add_argument("--gpu-n", type=int, default=0)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--method", type=str, choices=["time", "vtrain"])
    parser.add_argument("--log-path", type=str, default="./logs")
    args = parser.parse_args()
    
    input_size = args.input_size
    index_size = args.index_size
    dim_size = args.dim_size
    gpu_n = args.gpu_n
    test = args.test
    method = args.method
    log_path = args.log_path
    
    device = torch.device("cuda:%d" %gpu_n)
    
    if test:
        input_size = 6
        index_size = 3
        dim_size = 2
        
        input_tensor = torch.randn(input_size, dim_size, device=device)
        index_tensor = torch.randint(low=0, high=input_size, size=(index_size,), device=device)
        output_tensor = input_tensor[index_tensor]
        
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
        n_warmup = 20
        n_iter = 10
    
        final_time = 0
        
        if method == "vtrain":
            vp.init_trace()
            
        for i in range(n_warmup + n_iter):
            input_tensor_cpu = torch.randn(input_size, dim_size)
            assert input_tensor_cpu.dtype == torch.float
            input_tensor = input_tensor_cpu.to(device)
            
            index_tensor_cpu = torch.randint(low=0, high=input_size, size=(index_size,), dtype=torch.int32)
            assert index_tensor_cpu.dtype == torch.int32
            index_tensor = index_tensor_cpu.to(device)
            
            if i >= n_warmup:
                if method == "time":
                    start = time.time()
                
            output_tensor = input_tensor[index_tensor]
                
            if i >= n_warmup:
                if method == "time":
                    end = time.time()
                    final_time = final_time + end - start
                    
        if method == "time":
            final_time = final_time / n_iter * (10**6)
            
            print("number of inputs, number of indices, dim size, time (us)")
            print("%d, %d, %d, %f" %(input_size, index_size, dim_size, final_time))
        elif method == "vtrain":
            trace = vp.finish_trace().strip().split('\n')
            kernel_rows = []
            for i in range(len(trace)):
                row = trace[i]
                splitted_row = row.split(",")
                if splitted_row[2] == "KERNEL":
                    assert True
                    
                if splitted_row[2] == "KERNEL" and "index" in splitted_row[3]:
                    kernel_rows.append(float(splitted_row[1]))
                    
            assert len(kernel_rows) == n_warmup + n_iter
            kernel_rows_tensor = torch.tensor(kernel_rows)
            final_time = kernel_rows_tensor[-n_iter:].mean() / 1000
            
            print("number of inputs, number of indices, dim size, time (us)")
            print("%d, %d, %d, %f" %(input_size, index_size, dim_size, final_time))
            
        
if __name__ == "__main__":
    run()