from setuptools import find_packages, setup


with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name='medtool',
    packages=find_packages(),
    version='0.1.0',
    description='Project for easier training and preparing medical tools based on deep learning',
    author='Narek Maloyan',
    license='MIT',
)
