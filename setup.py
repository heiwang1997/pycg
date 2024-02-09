import re
from setuptools import find_packages, setup


with open("pycg/__init__.py", "r") as fh:
    pycg_version = re.findall(r'__version__ = \'(.*?)\'', fh.read())[0]


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='python-pycg',
    version=pycg_version,
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
    install_requires=[
        "numpy", "matplotlib",
        "pyquaternion", "pyyaml",
        "omegaconf", "tqdm",
        "pynvml", "calmsize",
        "rich", "plyfile",
        "trimesh", "imageio",
        "screeninfo", "pillow", "usd-core", "scipy"
    ],
    extras_require={
        "all": [
            "open3d_pycg_cpu"
        ]
    },
    include_package_data=True,
)
