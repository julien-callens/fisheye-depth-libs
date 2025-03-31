from setuptools import setup, find_packages

setup(
    name='functions',
    version='0.1.0',
    author='Your Name',
    description='A collection of utility functions for the lab.',
    packages=find_packages(),
    install_requires=[
        'plotly>=4.0.0',
        'matplotlib>=3.0.0',
        'pandas>=1.0.0',
        'numpy>=1.0.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
