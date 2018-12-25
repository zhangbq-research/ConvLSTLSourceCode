SPARNN
=======================================================================

[SPARNN](https://github.com/sxjscience/SPARNN), A Light-Weighted **Spatial Temporal RNN** Toolbox based on [Theano](http://deeplearning.net/software/theano/install.html)


###Requirements:

    Latest Python2 (python2.7.*)
    numpy + scipy
    pillow (Required if you want to generate your own moving-mnist dataset)
    Theano
    HDF5 + Netcdf + netcdf4-python* (Required if you want to use format of CURRENNT)
    ipython + matplotlib (Required if you want to view the demo)

###How to Install

1. SPARNN
   
   You can try:
   
            python setup.py install
   Or if you are not enough privilege:
   
            python setup.py install --user

2. Tips for installing python packages like "numpy + scipy, Theano, netcdf4-python"

   The easiest way to install all the required python packages is to use `pip install`.
   In case you find `pip` is not installed in your computer, you can simply get the latest version from https://pip.pypa.io/en/latest/installing.html .(It's recommended to download get_pip.py and use `python get_pip.py` or `python get_pip.py --user` if you aren't root)
   After that, you may install numpy + scipy, theano together with other packages by
   
            pip install numpy
            pip install scipy
            pip install Theano
            pip install pillow
            pip install netCDF4
    
   If you haven't got the root privilege, you can install these packages in the local user path by
            
            pip install numpy --user
            pip install scipy --user
            pip install Theano --user
            pip install pillow --user
            pip install netCDF4 --user
            
   [**IMPORTANT!**]For Theano, it's recommended to install the bleeding-edge version
   
            #Root level installation
            pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
            #User level installation        
            pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git --user
   
    
   [**IMPORTANT!**]Some further requirements are needed for these packages, the official installation guides may help a lot during your installation

3. Official Installation Guide For Related Packages
        
        Numpy & Scipy:
            http://docs.scipy.org/doc/numpy/user/install.html
        Theano:
            http://deeplearning.net/software/theano/install.html
        HDF5:
            https://hdfgroup.org/HDF5/
        Netcdf:
            http://www.unidata.ucar.edu/software/netcdf/
        netcdf4-python:
            https://github.com/Unidata/netcdf4-python

###How to Config Theano

Theano is the backbone of this project. To configure theano, view [theano-config](http://deeplearning.net/software/theano/library/config.html) for detailed help. You need to write the configuration to ~/.theanorc(or $HOME/.theanorc.txt for windows). The following is the recommended Theano configuration for SPARNN.

For CPU users:

    [global]
    floatX = float32
    device = cpu
    mode = FAST_RUN
    warn_float64 = warn

For GPU users(here the device can be any other GPU):

    [global]
    floatX = float32
    device = gpu0
    mode = FAST_RUN
    warn_float64 = warn

###Run Examples

To run the imdb sentiment analysis example

    python imdb_sentiment_analysis.py
    
To run the chime autoencoding example

    python chime_autoencoding.py

To run the mnist sequence forecasting example with a network of 2 layers of fully connected LSTM with 2048 nodes

(**IMPORTANT!**) Require large memory and computational resources, perhaps you should run it on GPU

    python mnist_sequence_forecasting_unconditional.py
      
To run the mnist sequence forecasting example with a 3 layer Convolutional-LSTM

(**IMPORTANT!**) It's also recommended to run this example on the GPU, although the parameter number is much smaller than the previous one

    python mnist_sequence_forecasting_conv_unconditional_deep.py
