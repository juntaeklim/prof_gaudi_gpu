from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gather_v_i32_indices',
    ext_modules=[
        CUDAExtension('custom_gather', [
            'gather_v_i32_indices.cpp',
            'gather_v_i32_indices_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })