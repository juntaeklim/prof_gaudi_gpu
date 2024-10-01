from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='scatter_v_i32_indices',
    ext_modules=[
        CUDAExtension('custom_scatter', [
            'scatter_v_i32_indices.cpp',
            'scatter_v_i32_indices_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })