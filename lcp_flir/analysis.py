"""
this library is intended for analysis and numerical calculations that are needed to work with FLIR and Point SDK

TODO:
- Move function that convert bytes stream to images with given pixel format
    - Supported Pixel Format: Mono12p, Mono16, Mono8, BayerRG12p, BayerRG16
"""

def convert_raw_mono12p_to_image(data, shape = None):
    """
    converts bytes stream of 12bit numbers to a 2D array representing an image

    see http://softwareservices.flir.com/BFS-U3-89S6/latest/Model/public/ImageFormatControl.html for details on how to parse the bytes stream.

    Parameters
    ----------
    data : 1D array
    shape : tuple

    Returns
    -------
    array
        2D array with given height(number of rows) and width(number of colums).
    """
    pass

def convert_raw_mono16_to_image(data, shape = None):
    """
    converts bytes stream of 16bit numbers to a 2D array representing an image

    see http://softwareservices.flir.com/BFS-U3-89S6/latest/Model/public/ImageFormatControl.html for details on how to parse the bytes stream.

    Parameters
    ----------
    data : 1D array
    shape : tuple

    Returns
    -------
    array
        2D array with given height(number of rows) and width(number of colums).
    """
    pass