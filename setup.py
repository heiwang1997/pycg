import sys
from setuptools import find_packages, setup
from pycg import __version__


setup(
    name='pycg',
    version=__version__,
    description='PyCG: Toolbox for CG-related visualizations and computations',
    author='Jiahui Huang',
    author_email='huangjh.work@outlook.com',
    keywords=['pycg', 'graphics', '3d', 'visualization'],
    python_requires='>=3.6',
    install_requires=[],
    packages=find_packages(),
    include_package_data=True,
)
