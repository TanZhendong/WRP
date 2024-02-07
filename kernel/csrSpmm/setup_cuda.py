from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='spmm_csr',
    ext_modules=[cpp_extension.CUDAExtension(
        'spmm_csr', ['csrSpmm.cpp']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)