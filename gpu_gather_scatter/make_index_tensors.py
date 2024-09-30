import torch
import argparse
import os

def run():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--type", type=str, choices=["linear", "power"])
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--entire", type=int, default=128)
    parser.add_argument("--gpu-n", type=int, default=0)
    parser.add_argument("--iter", type=int, default=20)
    parser.add_argument("--algo", type=str, choices=["randint", "randperm"])
    args = parser.parse_args()
    
    pattern = args.type
    start = args.start
    end = args.end
    stride = args.stride
    entire = args.entire
    gpu_n = args.gpu_n
    iter = args.iter
    algo = args.algo
    
    device = torch.device("cuda:%d" %gpu_n)

    if pattern == "linear":
        assert (end - start) % stride == 0
        index_sizes = [start + stride * i for i in range(int((end - start)/stride) + 1)]
    elif pattern == "power":
        index_sizes = []
        tmp = start
        while tmp <= end:
            index_sizes.append(tmp)
            tmp = tmp * stride
        assert tmp / stride == end
    
    for index_size in index_sizes:
        print("index_size: %d" %index_size)
        for i in range(iter):
            file_path = "./index_tensors/%s_entire_%d_index_%d_iter_%d" % (algo, entire, index_size, i)
            
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping...")
                continue
            
            if algo == "randint":
                index_tensor = torch.randint(low=0, high=entire, size=(index_size,), dtype=torch.int32, device = device)
                index_tensor = index_tensor.to("cpu")
                torch.save(index_tensor, file_path)
            elif algo == "randperm":
                index_tensor = torch.randperm(entire, device=device)[:index_size]
                index_tensor = index_tensor.to("cpu")
                torch.save(index_tensor, file_path)
            else:
                assert False

    
if __name__ == "__main__":
    run()