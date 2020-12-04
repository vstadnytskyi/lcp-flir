from time import time, sleep
from numpy import zeros, right_shift, array
import PySpin
from PySpin import System

import platform

if 0:
    import matplotlib
    if platform.system() == 'Windows':
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('WxAgg')
    from matplotlib import pyplot as plt
    plt.ion()
import os
#os.nice(-20)


def start():
    camera_dm34.start_thread()
    camera_dm4.start_thread()
    camera_dm16.start_thread()

def init_recording(comments = '', N_frames = 1200):
    camera_dm34.recording_init(comments = comments, N_frames = N_frames)
    camera_dm4.recording_init(comments = comments, N_frames = N_frames)
    camera_dm16.recording_init(comments = comments, N_frames = N_frames)

def init_recording_one_file(comments = '', N_frames = 600):
    """
    """
    from h5py import File
    filename = '/mnt/data/file_' + comments + '.hdf5'
    lst = ['dm16','dm34','dm4']
    with File(filename,'a') as f:
        for item in lst:
            f.create_dataset(f'{item}/exposure time', data = 'trigger width')
            f.create_dataset(f'{item}/black level all', data = self.black_level['all'])
            f.create_dataset(f'{item}/black level analog', data = self.black_level['analog'])
            f.create_dataset(f'{item}/black level digital', data = self.black_level['digital'])
            f.create_dataset(f'{item}/gain', data = self.gain)
            f.create_dataset(f'{item}/time', data = time())
            f.create_dataset(f'{item}/temperature', data = self.temperature)
            f.create_dataset(f'{item}/images', (N_frames,3000,4096), dtype = 'int16', chunks = (1,3000,4096))
            f.create_dataset('timestamps', (N_frames,) , dtype = 'float64')

def start_recording():
    camera_dm34.recording_start()
    camera_dm4.recording_start()
    camera_dm16.recording_start()

def stop_recording():
    camera_dm34.recording_stop()
    camera_dm4.recording_stop()
    camera_dm16.recording_stop()

def record_sequential(comments = '', N_frames = 600, M_chunks = 2):
    """
    """
    from time import sleep, ctime, time

    def sync():
        reset()
        resume()
        while not is_synced()[0]:
            reset()
            sleep(1/15.625)
    for M_i in range(M_chunks):
        sync()
        print(ctime(time()),is_synced())
        sleep((N_frames+5)/15.625)
        pause()
        t1 = time()
        print(ctime(time()),'Saving to the drive')
        init_recording(comments = comments + f'_{M_i}', N_frames=N_frames)
        camera_dm34.recording = True
        camera_dm34.recording_run()

        camera_dm16.recording = True
        camera_dm16.recording_run()

        camera_dm4.recording = True
        camera_dm4.recording_run()
        t2 = time()
        resume()
        print(ctime(time()),f'time to save to the drive: {(t2-t1)/N_frames} per frame')

def stop():
    from ubcs_auxiliary.threading import new_thread
    new_thread(camera_dm4.stop_thread)
    new_thread(camera_dm16.stop_thread)
    new_thread(camera_dm34.stop_thread)

def pause():
    from ubcs_auxiliary.threading import new_thread
    new_thread(camera_dm34.pause_acquisition)
    new_thread(camera_dm4.pause_acquisition)
    new_thread(camera_dm16.pause_acquisition)

def resume():
    from ubcs_auxiliary.threading import new_thread
    new_thread(camera_dm34.resume_acquisition)
    new_thread(camera_dm4.resume_acquisition)
    new_thread(camera_dm16.resume_acquisition)

def get_timestamps():
    result = {}
    for key in list(cameras.keys()):
        result[key] = cameras[key].extract_timestamp_image(cameras[key].queue.peek_front())
    return result

def is_synced(dt = 0.054):
    """
    """
    mdt = get_timestamps()
    from numpy import zeros
    arr = zeros((3,3))
    lst = list(mdt.keys())
    for i in range(3):
        for j in range(3):
            arr[i,j] = abs(mdt[lst[i]]-mdt[lst[j]])

    return (arr < dt).all(), arr, lst

def close():
    stop()
    del camera_dm34.cam
    del camera_dm16.cam
    del camera_dm4.cam
    system.ReleaseInstance()

def reset():
    camera_dm4.queue.reset()
    camera_dm16.queue.reset()
    camera_dm34.queue.reset()

def start_recording_chunks(N = 600, N_chunks = 3):
    from time import time
    for i in range(N_chunks):
        t = time()
        print(f'Chunk {i}')
        print(f'reset and sleep: {time()-t}')
        reset()
        sleep(600/19)
        print(f'start recording  : {time()-t}')
        start_recording()
        while (camera_dm34.recording_pointer <= 600*(i+1)) and (camera_dm16.recording_pointer <= 600*(i+1)) and (camera_dm4.recording_pointer <= 600*(i+1)):
            sleep(0.05)
            if (camera_dm34.recording_pointer == 600*(N_chunks)) and (camera_dm16.recording_pointer == 600*(N_chunks)) and (camera_dm4.recording_pointer == 600*(N_chunks)):
                break
        print(f'stop recording  : {time()-t}')
        stop_recording()
    print(f'Chunks recording is Done recording')

def set_gain(gain):
    camera_dm34.gain = gain
    camera_dm16.gain = gain
    camera_dm4.gain = gain
def get_gain():
    dic = {}
    dic['dm34'] = camera_dm34.gain
    dic['dm16'] = camera_dm16.gain
    dic['dm4'] = camera_dm4.gain
    return dic

def set_black(value):
    camera_dm34.black_level = value
    sleep(0.1)
    camera_dm16.black_level = value
    sleep(0.1)
    camera_dm4.black_level = value
    sleep(0.1)

def get_black():
    dic = {}
    dic['dm34'] = camera_dm34.black_level
    sleep(0.1)
    dic['dm16'] = camera_dm16.black_level
    sleep(0.1)
    dic['dm4'] = camera_dm4.black_level
    sleep(0.1)
    return dic

def set_exposure(value):
    camera_dm34.exposure_time = value
    camera_dm16.exposure_time = value
    camera_dm4.exposure_time = value
def get_exposure():
    dic = {}
    dic['dm34'] = camera_dm34.exposure_time
    dic['dm16'] = camera_dm16.exposure_time
    dic['dm4'] = camera_dm4.exposure_time
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

def get_temperature():
    import subprocess
    cmd = "sudo nvme smart-log /dev/nvme0 | grep '^temperature'"
    output = subprocess.check_output(cmd, shell=True)
    return int(str(output).split('C')[0].split(':')[1])


def save_buffer(root, comments):
    from ubcs_auxiliary.threading import new_thread
    lst = [camera_dm34,camera_dm16,camera_dm4]
    print('started saving to a drive')
    for item in lst:
        item.recording_filename = root + item.name +'_'+ comments + '.hdf5'
    new_thread(camera_dm34.record_once,600)
    new_thread(camera_dm16.record_once,600)
    camera_dm4.record_once(600)
    print('done saving to a drive')

if __name__ == '__main__':
    print('Starting 3 Cameras Code')
    from lcp_video.flir_camera.flir_camera_DL import FlirCamera

    system = System.GetInstance()
    camera = FlirCamera('test',system)
    camera.get_all_cameras()
    root = '/mnt/data/'
    camera_dm4 = FlirCamera('dm4',system)
    camera_dm4.init(serial_number = '19490369', settings = 1)
    camera_dm4.recording_root = root

    camera_dm34 = FlirCamera('dm34', system)
    camera_dm34.init(serial_number = '18159480', settings = 1)
    camera_dm34.recording_root = root

    camera_dm16 = FlirCamera('dm16', system)
    camera_dm16.init(serial_number = '18159488', settings = 1)
    camera_dm16.recording_root = root

    cameras = {'dm34':camera_dm34,'dm16':camera_dm16,'dm4':camera_dm4}

    start()
