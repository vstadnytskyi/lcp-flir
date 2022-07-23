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


def extract_timestamp_image(raw,img_len):
    """
    Extracts header information from the

    returns timestamp_lab,timestamp_cam,frameid
    """
    from lcp_flir.analysis import bin_array, binarr_to_number
    pointer = img_len
    header_img = raw[pointer:pointer+64]
    timestamp_lab =  binarr_to_number(header_img)/1000000
    header_img = raw[pointer+64:pointer+128]
    timestamp_cam =  binarr_to_number(header_img)
    header_img = raw[pointer+128:pointer+192]
    frameid =  binarr_to_number(header_img)
    return timestamp_lab,timestamp_cam,frameid

def binarr_to_number(vector):
    """
    converts a vector of bits into an integer.
    """
    num = 0
    from numpy import flip
    vector = flip(vector)
    length = vector.shape[0]
    for i in range(length):
        num += (2**(i))*vector[i]
    return num

def bin_array(num, m):
    """
    Converts a positive integer num into an m-bit bit vector
    """
    from numpy import uint8, binary_repr,array
    return array(list(binary_repr(num).zfill(m))).astype(uint8)