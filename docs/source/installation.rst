============
Installation
============



This library can be installed from your local location on your computer. First, you would need to clone the repository in desired location. Second, run the following command at the command line::

    $ pip install -e .
    

Currently, this library cannot be installed from PyPi but will be available in future. Please stay tuned...
To install from PyPi at the command line::

    $ pip install lcp-flir

To check if the library is correctly installed open python or ipython shell at the command line::
    
    $ ipython
    $ import lcp_flir
    $ lcp_flir.__version__
    $ lcp_flir.__file__ #The last command should show you location of the folder on your harddrive. Confirm it is correct location.


FLIR cameras require FLIR SDK which is called spinnaker. It can be donwloaded from FLIR webpage https://www.flir.com/products/spinnaker-sdk or there is a copy in this library /lcp-flir/flir-spinnaker/spinnaker_python-2.0.0.109-cp37-cp37m-linux_x86_64.whl

If you have Linux, you can install spinnaker from wheel, run following command at the command line::

    $ python -m wheel install flir-spinnaker/spinnaker_python-2.0.0.109-cp37-cp37m-linux_x86_64.whl
