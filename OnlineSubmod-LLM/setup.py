from setuptools import setup
import setuptools
import os

long_description = "CORDS is COReset and Data Selection library for making machine learning time, energy, cost, and compute efficient. CORDS is built on top of PyTorch. Today, deep learning systems are extremely compute-intensive, with significant turnaround times, energy inefficiencies, higher costs, and resource requirements [7, 8]. CORDS is an effort to make deep learning more energy, cost, resource, and time-efficient while not sacrificing accuracy."

setup(
    name='cords',
    version='v0.0.4',
    author='Krishnateja Killamsetty, Dheeraj Bhat, Rishabh Iyer',
    author_email='krishnatejakillamsetty@gmail.com',
    #packages=['cords', 'cords/selectionstrategies', 'cords/utils'],
    url='XXXX',
    license='LICENSE.txt',
    packages=setuptools.find_packages(),
    description='cords is a package for data subset selection for efficient and robust machine learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        
            ],
)
