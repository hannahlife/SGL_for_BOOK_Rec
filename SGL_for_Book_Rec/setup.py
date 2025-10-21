from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

# Cython扩展
extensions = [
    Extension(
        "reckit.cython.eval_matrix",
        ["reckit/cython/eval_matrix.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "reckit.cython.randint_choice", 
        ["reckit/cython/randint_choice.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="SGL-Torch",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    install_requires=[
        "numpy",
        "scipy", 
        "pandas",
        "cython"
    ]
)
