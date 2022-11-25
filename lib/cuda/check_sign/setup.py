from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='check_sign',
    ext_modules=[
        CUDAExtension('check_sign', [
            'mesh_intersection_cudae_cuda.cpp',
            'mesh_intersection_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
