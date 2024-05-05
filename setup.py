from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.4.1"
DESCRIPTION = "Building Vision models in Keras3 for framework agnostic training and inference."

# Setting up
setup(
    name="keras-vision",
    version=__version__,
    author="Vaibhav Singh",
    author_email="vaibhav.singh.3001@gmail.com",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/veb-101/keras-vision",
    # packages=find_packages(exclude=["tests"]),
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # Note that this is a string of words separated by whitespace, not a list.
    keywords="keras3 tensorflow Jax PyTorch Vision",
    install_requires=[
        "keras",
    ],
    python_requires=">=3.10",
    license="Apache 2.0",
)
