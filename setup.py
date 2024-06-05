from setuptools import setup, find_packages

setup(
    name='RTfitting',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy', 'pandas', 'latex', 'matplotlib'
    ],
    author='Peipei Hao',
    author_email='peipei.hao@colorado.edu',
    description='Fitting tools for the temperature-dependent resistivity data',
    url='https://github.com/peipeihao/RTfitting2',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
