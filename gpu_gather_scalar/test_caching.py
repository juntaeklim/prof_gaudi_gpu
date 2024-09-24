import torch
import argparse
import time
import vtrain_profiler as vp

def run():
    device = torch.device("cuda:0")
    
    for i in range(3):
        a = torch.rand(8)
        a_gpu = a.to(device)
        torch.cuda.empty_cache()
        b_gpu = 2 * a_gpu
        
        
if __name__ == "__main__":
    run()