from setuptools import setup, find_packages

setup(
    name='pyRMT',
    version='0.1',
    description='Reference Map Technique utilities',
    author='Saman Seifi',
    packages=find_packages(),  # Automatically finds pyRMT/
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'h5py',
        'pyamg',
        'numba'
    ],
    include_package_data=True,
    zip_safe=False,
)
