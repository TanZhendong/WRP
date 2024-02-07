from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='spmm_block_ell',
    ext_modules=[cpp_extension.CUDAExtension(
        'spmm_block_ell', ['blockSpmm.cpp']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)