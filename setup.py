from setuptools import find_packages, setup
from pycg import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='python-pycg',
    version=__version__,
    author='Jiahui Huang',
    author_email='huangjh.work@outlook.com',
    description='PyCG: Toolbox for CG-related visualizations and computations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/heiwang1997/pycg",
    packages=find_packages(),
    classifiers=[
        "Operating System :: Unix",
        "Topic :: Multimedia :: Graphics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only"
    ],
    keywords=['pycg', 'graphics', '3d', 'visualization'],
    python_requires='>=3.6',
    install_requires=[],
    include_package_data=True,
)
