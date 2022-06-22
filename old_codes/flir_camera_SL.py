from time import time, sleep
from numpy import zeros, right_shift, array
import PySpin
from PySpin import System
from matplotlib import pyplot as plt
plt.ion()

class FLIR_SL():
    def __init__(self):
        self.recording_dir = self.get_tempdir()

    def init(self):
        pass

    def start(self):
        """
        starts data acquisition threads
        """

        camera_12mm.start_thread()
        camera_tc.start_thread()
        camera_8mm.start_thread()

    def stop(self):
        """
        stops data acquisition threads
        """
        from ubcs_auxiliary.threading import new_thread
        new_thread(camera_tc.stop_thread)
        new_thread(camera_8mm.stop_thread)
        new_thread(camera_12mm.stop_thread)

    def run_once_recording(self, N, filename):
        from h5py import File
        if self.cam.queue.length > 1:
            image = self.cam.queue.dequeue(1)
            with File(filename,'a') as f:
                t = extract_timestamp_image(image)
                f.create_dataset(str(i),image)

    def run_recording(self):
        pass

    def start_recording(self):
        from ubcs_auxiliary.threading import new_thread
        new_thread(self.run_recording)

    def get_tempdir(self):
        from tempfile import gettempdir
        return gettempdir()

    def save_buffer(self, root = '', label = ''):
        from tempfile import gettempdir
        if root == '':
            root = gettempdir()

def extract_header(arr):
    """
    Takes array of images (length, rows, cols) and returns the header data

    The header information is located in the last row (row = 3000) of an image
    """
    header_arr = arr[:,-1,:]
    length = header_arr.shape[0]
    dic = {}
    dic['timestamp'] = zeros((length,))
    for i in range(length):
        dic['timestamp'][i] =  binarr_to_number(header_arr[i,:64])/1000000
    return dic

def extract_timestamp_image(arr):
    """
    Takes array of images (length, rows, cols) and returns the header data

    The header information is located in the last row (row = 3000) of an image
    """
    header_arr = arr[-1,:64]
    length = header_arr.shape[0]
    time_stamp =  binarr_to_number(header_arr)/1000000
    return time_stamp

def binarr_to_number(vector):
    num = 0
    from numpy import flip
    vector = flip(vector)
    length = vector.shape[0]
    for i in range(length):
       num += (2**(i))*vector[i]
    return num

def bin_array(num, m):
    from numpy import uint8, binary_repr,array
    """Convert a positive integer num into an m-bit bit vector"""
    return array(list(binary_repr(num).zfill(m))).astype(uint8)

if __name__ == '__main__':
    from lcp_video.flir_camera.flir_camera_DL import FlirCamera

    system = System.GetInstance()
    camera = FlirCamera('test',system)
    camera.get_all_cameras()

    camera_tc = FlirCamera('telecentric',system)
    camera_tc.init(serial_number = '19490369', settings = 1)

    camera_12mm = FlirCamera('12 mm', system)
    camera_12mm.init(serial_number = '18159480', settings = 1)

    camera_8mm = FlirCamera('8 mm', system)
    camera_8mm.init(serial_number = '18159488', settings = 1)

    #camera_12mm.start_thread()
    #camera_tc.start_thread()
    #camera_8mm.start_thread()
