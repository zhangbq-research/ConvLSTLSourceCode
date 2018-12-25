import os
import setuptools


setuptools.setup(
    name='SPARNN',
    version="0.1.dev0",
    author="SHI Xingjian",
    author_email="sxjscience001@gmail.com",
    packages=setuptools.find_packages(),
    description='A Light-Weighted Spatial Temporal RNN Toolbox based on Theano',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.md')).read(),
    license='MIT',
    url='https://github.com/sxjscience/SPARNN',
    install_requires=['numpy', 'scipy', 'pillow', 'netCDF4', 'Theano'],
    dependency_links=['git+http://github.com/Theano/Theano.git#egg=Theano', ],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
