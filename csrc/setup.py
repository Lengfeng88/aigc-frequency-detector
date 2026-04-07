# csrc/setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Get PyTorch's CUDA configuration
torch_cuda_version = torch.version.cuda
print(f"Using PyTorch CUDA version: {torch_cuda_version}")

# Get include and library paths from PyTorch
include_dirs = torch.utils.cpp_extension.include_paths()
library_dirs = torch.utils.cpp_extension.library_paths()

# Add CUDA include path if available
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
if os.path.exists(cuda_home):
    include_dirs.append(os.path.join(cuda_home, 'include'))
    library_dirs.append(os.path.join(cuda_home, 'lib64'))

setup(
    name='gradient_variance',
    ext_modules=[
        CUDAExtension(
            name='gradient_variance_cuda',
            sources=[
                'gradient_variance.cpp',
                'gradient_variance_cuda.cu',
                'gradient_variance_cpu.cpp'
               
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3', 
                    '--use_fast_math',
                    f'--compiler-options=-fPIC'
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)},
    zip_safe=False
)