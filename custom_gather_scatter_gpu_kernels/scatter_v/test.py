import torch
from custom_scatter import scatter_v_i32_indices

device = torch.device("cuda:0")

a = torch.arange(1024 * 32).view(-1, 32).to(torch.float)
b = torch.randint(1024*1024, size=(1024,), dtype=torch.int32)
a_gpu = a.to(device)
b_gpu = b.to(device)
out_gpu = torch.zeros(1024*1024*32, device=device).view(-1, 32).to(torch.float)

out_cpu = scatter_v_i32_indices(a_gpu, b_gpu, out_gpu).to("cpu")

c = torch.zeros(1024*1024*32).view(-1, 32).to(torch.float)
c[b] = c[b] + a
print(torch.all(out_cpu == c))