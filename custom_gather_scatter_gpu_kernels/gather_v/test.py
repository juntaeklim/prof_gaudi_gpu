import torch
from custom_gather import gather_v_i32_indices

device = torch.device("cuda:0")

a = torch.arange(1024*1024*32).view(-1, 32).to(torch.float)
b = torch.randint(1024*1024, size=(1024,), dtype=torch.int32)
a_gpu = a.to(device)
b_gpu = b.to(device)

out = gather_v_i32_indices(a_gpu, b_gpu)
out_cpu = out.to("cpu")

print(torch.all(out_cpu == a[b]))