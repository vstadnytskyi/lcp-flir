
from time import time, sleep
from numpy import zeros, right_shift, array, copy
import PySpin

from PySpin import PixelFormat_Mono16,PixelFormat_Mono12Packed, PixelFormat_Mono12p

class FlirCamera():

    def __init__(self, name = None, system = None):
        from numpy import zeros, nan
        self.name = name
        self.system = system
        self.last_image_time = zeros((1,20))
        self.recording_filename = f'camera_{name}.hdf5'
        self.recording_root = '/mnt/data/'
        self.recording_N = 1
        self.recording = False
        self.recording_pointer = 0
        self.threads = {}

    def init(self, serial_number,settings = 1):
        from numpy import zeros, nan
        print('')
        print('-------------INITIALIZING NEW CAMERA-----------')
        import PySpin
        from circular_buffer_numpy.queue import Queue
        from circular_buffer_numpy.circular_buffer import CircularBuffer
        self.acquiring = False

        self.cam = self.find_camera(serial_number = serial_number)
        self.nodes = self.get_nodes()

        self.height = self.get_height()
        self.width = self.get_width()
        self.queue = Queue((620,self.height+1,self.width), dtype = 'uint16')
        self.analysis_buffer = CircularBuffer(shape = (100,20), dtype = 'float64')
        self.last_image_time_buffer = CircularBuffer(shape = (6000,20), dtype = 'float64')
        self.last_image_time_buffer.buffer = self.last_image_time_buffer.buffer*nan
        self.timestamp_buffer = CircularBuffer(shape = (6000,1), dtype = 'float64')
        self.last_image = zeros((self.height+1,self.width), dtype = 'uint16')
        self.set_lut_enable(False)
        self.set_gamma_enable(False)
        #self.set_gamma(0.5)
        self.set_autoexposure('Off')
        self.set_autogain('Off')

        self.configure_transport()
        try:
            self.cam.AcquisitionStop()
            print ("Acquisition stopped")
        except PySpin.SpinnakerException as ex:
            print("Acquisition was already stopped")

        print('setting pixel format Mono11Packed')
        self.cam.PixelFormat.SetValue(PixelFormat_Mono12p)

        self.set_exposure_mode('Timed')
        self.exposure_time = 63000 #53500
        self.gain = 0
        self.black_level = 15
        self.background = zeros((self.height+1,self.width))
        self.background[:self.height,:] = 15
        self.background_flag = True

        self.conf_acq_and_trigger(settings=settings)

    def close(sself):
        pass

    def kill(self):
        """
        """
        self.pause_acquisition()
        sleep(1)
        self.stop_acquisition()
        sleep(0.1)
        del self.cam
        self.system.ReleaseInstance()
        del self

    def reset_to_factory_settings(self):
        self.pause_acquisition()
        self.stop_acquisition()
        try:
            self.cam.AcquisitionStop()
            print ("Acquisition stopped")
        except PySpin.SpinnakerException as ex:
            print("Acquisition was already stopped")
        self.cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
        self.cam.UserSetLoad()

    def configure_transport(self):
        cam = self.cam
        # Configure Transport Layer Properties
        cam.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_OldestFirst )
        cam.TLStream.StreamBufferCountMode.SetValue(PySpin.StreamBufferCountMode_Manual )
        cam.TLStream.StreamBufferCountManual.SetValue(10)
        print(f"Buffer Handling Mode: {cam.TLStream.StreamBufferHandlingMode.GetCurrentEntry().GetSymbolic()}")
        print( f"Buffer Count Mode: {cam.TLStream.StreamBufferCountMode.GetCurrentEntry().GetSymbolic()}")
        print(f"Buffer Count: {cam.TLStream.StreamBufferCountManual.GetValue()}")
        print(f"Max Buffer Count: {cam.TLStream.StreamBufferCountManual.GetMax()}")

    def deinit(self):
        self.stop_acquisition()
        self.cam.DeInit()


    def find_camera(self, serial_number = None):
        """
        looks for all cameras connected and returns cam object
        """
        cam = None
        cam_list = self.system.GetCameras()
        num_cameras = cam_list.GetSize()
        print(f'found {num_cameras} cameras')
        for i,cam in enumerate(cam_list):
            sn = cam.TLDevice.DeviceSerialNumber.GetValue()
            print(f'sn = {sn}')
            if serial_number == sn:
                break
        cam_list.Clear()
        return cam

    def get_all_cameras(self):
        cam_list = self.system.GetCameras()
        num_cameras = cam_list.GetSize()
        print(f'found {num_cameras} cameras')
        cameras = []
        for i,cam in enumerate(cam_list):
            sn = cam.TLDevice.DeviceSerialNumber.GetValue()
            print(f'sn = {sn}')
            cameras.append(sn)
        cam_list.Clear()
        return cameras

    def get_nodes(self):
        import PySpin
        self.cam.Init()
        # Retrieve GenICam nodemap
        nodemap = self.cam.GetNodeMap()
        self.nodemap = nodemap
        nodes = {}
        nodes['auto_gain'] = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        nodes['pixel_format'] = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
        nodes['gain'] = PySpin.CEnumerationPtr(nodemap.GetNode('Gain'))
        nodes['acquisition_mode'] = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        nodes['exposure_time'] = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureTime'))
        return nodes


    def get_background(self):
        from numpy import mean
        self.background = mean(self.queue.buffer, axis = 0)

    def get_image(self):
        from numpy import right_shift, nan, zeros
        self.last_image *= 0
        if self.acquiring:
            self.last_image_time[0,3] = time()
            image_result = self.cam.GetNextImage()
            timestamp = image_result.GetTimeStamp()
            self.last_image_result = image_result
            self.last_image_time[0,4] = time()
            # Getting the image data as a numpy array
            image_data = image_result.Convert(PixelFormat_Mono16).GetNDArray()
            image_data = right_shift(image_data,4)
            self.last_image_time[0,5] = time()
            image_result.Release()
            self.last_image_time[0,6] = time()
        else:
            print('No Data in get image')
            image_data = zeros((self.height,self.width))
        self.last_image_time[0,7] = time()
        self.last_image[0:self.height,:] = image_data
        self.last_image[self.height,:64] = self.get_image_header(value =int(time()*1000000), length = 64)
        self.last_image[self.height,64:128] = self.get_image_header(value = timestamp, length = 64)
        self.last_image_time[0,8] = time()
        return self.last_image

    def get_image_header(self,value = None, length = 4096):
        from time import time
        from numpy import zeros
        from ubcs_auxiliary.numerical import bin_array
        arr = zeros((1,length))
        if value is None:
            t = int(time()*1000000)
        else:
            t = value
        arr[0,:64] = bin_array(t,64)
        return arr

    def binarr_to_number(self,arr):
        num = 0
        from numpy import flip
        arr = flip(arr)
        length = arr.shape[0]
        for i in range(length):
           num += (2**(i))*arr[i]
        return num

    def bin_array(num, m):
        from numpy import uint8, binary_repr,array
        """Convert a positive integer num into an m-bit bit vector"""
        return array(list(binary_repr(num).zfill(m))).astype(uint8)

    def run_once(self):
        if self.acquiring:
            img = self.get_image()
            self.last_image_time[0,9] = time()
            data = img.reshape(1,img.shape[0],img.shape[1])
            self.last_image_time[0,10] = time()
            self.queue.enqueue(data)
            self.last_image_time[0,11] = time()
            self.last_image_time_buffer.append(self.last_image_time)
            self.last_image_time[0,12] = time()
    def run(self):
        while self.acquiring:
            self.run_once()


    def start_thread(self):
        from ubcs_auxiliary.multithreading import new_thread
        if not self.acquiring:
            self.start_acquisition()
            self.threads['acquisition'] = new_thread(self.run)

    def stop_thread(self):
        if self.acquiring:
            self.stop_acquisition()

    def resume_acquisition(self):
        """
        """
        from ubcs_auxiliary.multithreading import new_thread
        if not self.acquiring:
            self.acquiring = True
            self.threads['acquisition'] = new_thread(self.run)

    def pause_acquisition(self):
        """
        """
        self.acquiring = False



    def start_acquisition(self):
        """
        a wrapper to start acquisition of images.
        """
        self.acquiring = True
        self.cam.BeginAcquisition()
    def stop_acquisition(self):
        """
        a wrapper to stop acquisition of images.
        """
        self.acquiring = False
        try:
            self.cam.EndAcquisition()
            print ("Acquisition ended")
        except PySpin.SpinnakerException as ex:
            print("Acquisition was already ended")


    def get_black_level(self):
        self.cam.BlackLevelSelector.SetValue(0)
        all = self.cam.BlackLevel.GetValue()*4095/100
        self.cam.BlackLevelSelector.SetValue(1)
        analog = self.cam.BlackLevel.GetValue()*4095/100
        self.cam.BlackLevelSelector.SetValue(2)
        digital = self.cam.BlackLevel.GetValue()*4095/100
        self.cam.BlackLevelSelector.SetValue(0)
        return {'all':all,'analog':analog,'digital':digital}

    def set_black_level(self,value):
        """
        """
        self.cam.BlackLevelSelector.SetValue(0)
        self.cam.BlackLevel.SetValue(value*100/4095)
    black_level = property(get_black_level,set_black_level)

    def get_temperature(self):
        """
        """
        temp = self.cam.DeviceTemperature.GetValue()
        return temp
    temperature = property(get_temperature)

    def get_fps():
        return nan
    def set_fps(value):
        pass
    fps = property(get_fps,set_fps)

    def get_gain(self):
        return self.cam.Gain.GetValue()
    def set_gain(self,value):
        if value >= self.cam.Gain.GetMax():
            value = self.cam.Gain.GetMax()
        elif value < self.cam.Gain.GetMin() :
            value = self.cam.Gain.GetMin()
        self.cam.Gain.SetValue(value)
    gain = property(get_gain,set_gain)

    def get_autogain(self):
        if self.cam is not None:
            value = self.cam.GainAuto.GetValue()
            if value == 0:
                return 'Off'
            elif value == 1:
                return 'Once'
            elif value == 2:
                return 'Continuous'
        else:
            return None
    def set_autogain(self,value):
        node = self.nodes['auto_gain']
        off = node.GetEntryByName("Off")
        once = node.GetEntryByName("Once")
        continuous = node.GetEntryByName("Continuous")
        if value == 'Off':
            node.SetIntValue(off.GetValue())
        elif value == 'Once':
            node.SetIntValue(once.GetValue())
        elif value == 'Continuous':
            node.SetIntValue(continuous.GetValue())
    autogain = property(get_autogain,set_autogain)

    def get_serial_number(self):
        import PySpin
        device_serial_number = PySpin.CStringPtr(self.nodemap_tldevice.GetNode('DeviceSerialNumber'))
        return device_serial_number

    def get_acquisition_mode(self):
        node = self.nodes['acquisition_mode']

    def set_acquisition_mode(self,value):
        node = self.nodes['acquisition_mode']
        continious = node.GetEntryByName("Continuous")
        single_frame = node.GetEntryByName("SingleFrame")
        multi_frame = node.GetEntryByName("MultiFrame")
        if value == 'Continuous':
            mode = continious
        elif value == 'SingleFrame':
            mode = single_frame
        elif value == 'MultiFrame':
            mode = multi_frame
        print(f'setting acquisition mode {value}')
        self.stop_acquisition()
        node.SetIntValue(mode)
        self.start_acquisition()

    def get_lut_enable(self):
        """
        """
        if self.cam is not None:
            result = self.cam.LUTEnable.GetValue()
        else:
            result = None
        return result
    def set_lut_enable(self,value):
        """
        """
        if self.cam is not None:
            self.cam.LUTEnable.SetValue(value)
        print(f'setting LUT enable {value}')
    lut_enable = property(get_lut_enable,set_lut_enable)

    def get_gamma_enable(self):
        """
        """
        if self.cam is not None:
            result = self.cam.GammaEnable.GetValue()
        else:
            result = None
        return result
    def set_gamma_enable(self,value):
        """
        """
        if self.cam is not None:
            self.cam.GammaEnable.SetValue(value)
        print(f'setting gamma enable {value}')
    gamma_enable = property(get_gamma_enable,set_gamma_enable)

    def get_gamma(self):
        if self.cam is not None:
            result = None
            #result = self.cam.Gamma.GetValue()
        else:
            result = None
        return result
    def set_gamma(self,value):
        """
        """
        if self.cam is not None:
            result = None
            #self.cam.Gamma.SetValue(value)
        print(f'setting gamma {value}')
    gamma = property(get_gamma,set_gamma)

    def get_height(self):
        if self.cam is not None:
            reply = self.cam.Height.GetValue()
        else:
            reply = nan
        return reply

    def get_width(self):
        if self.cam is not None:
            reply = self.cam.Width.GetValue()
        else:
            reply = nan
        return reply

    def set_exposure_mode(self,mode = None):
        from PySpin import ExposureMode_Timed, ExposureMode_TriggerWidth
        print(f'Setting up ExposureMode to {mode}')
        if self.cam is not None:
            if mode == "Timed":
                self.cam.ExposureMode.SetValue(ExposureMode_Timed)
            elif mode == "TriggerWidth":
                self.cam.ExposureMode.SetValue(ExposureMode_TriggerWidth)

    def set_autoexposure(self, value = 'Off'):
        from PySpin import ExposureAuto_Off, ExposureAuto_Continuous, ExposureAuto_Once
        if value == 'Off':
            self.cam.ExposureAuto.SetValue(ExposureAuto_Off)
        elif value == 'Once':
            self.cam.ExposureAuto.SetValue(ExposureAuto_Once)
        elif value == 'Continuous':
            self.cam.ExposureAuto.SetValue(ExposureAuto_Continuous)
            print(f'setting gamma enable {value}')
    def get_autoexposure(self, bool = False):
        value = self.cam.ExposureAuto.GetValue()
        if value == 0:
            return 'Off'
        elif value == 1:
            return 'Once'
        elif value == 2:
            return 'Continuous'
        else:
            return 'unknown'

    def set_exposure_time(self, value):
        if self.cam is not None:
            self.cam.ExposureTime.SetValue(value)
            self._exposure_time = value
        else:
            pass
    def get_exposure_time(self,):
        if self.cam is not None:
            return self.cam.ExposureTime.GetValue()
        else:
            return nan
        self._exposure_time = value
    exposure_time = property(get_exposure_time,set_exposure_time)

    def set_pixel_format(self, value = 'Mono16'):
        self.cam.PixelFormat.SetIn

##### Acquisition Control
#####


    def trigger_now(self):
        """
        software trigger for camera
        """
        import PySpin
        self.cam.TriggerSoftware()

    def conf_acq_and_trigger(self, settings = 1):
        """
        a collection of setting defined as appropriate for the selected modes of operation.

        settings 1 designed for external trigger mode of operation
        Acquisition Mode = Continuous
        Acquisition FrameRate Enable = False
        LUTEnable = False
        TriggerSource = Line 0
        TriggerMode = On
        TriggerSelector = FrameStart
        """
        import PySpin
        if self.cam is not None:
            if settings ==1:
                print('Acquisition and Trigger Settings: 1')
                print('setting continuous acquisition mode')
                self.cam.AcquisitionMode.SetIntValue(PySpin.AcquisitionMode_Continuous)
                print('setting frame rate enable to False')
                self.cam.AcquisitionFrameRateEnable.SetValue(False)

                print('setting Look Up Table Enable to False')
                self.cam.LUTEnable.SetValue(False)

                print('setting TriggerSource Line0')
                self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
                print('setting TriggerMode On')
                self.cam.TriggerMode.SetIntValue(PySpin.TriggerMode_On)

                print('setting TriggerSelector FrameStart')
                self.cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
                print('setting TriggerActivation RisingEdge')
                self.cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
                print('setting TriggerOverlap ReadOnly ')
                self.cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)


            elif settings ==2:
                print('Acquisition and Trigger Settings: 2')
                print('setting SingleFrame acquisition mode')
                self.cam.AcquisitionMode.SetIntValue(PySpin.AcquisitionMode_SingleFrame)
                print('setting frame rate enable to False')
                self.cam.AcquisitionFrameRateEnable.SetValue(False)
                print('setting Look Up Table Enable to False')
                self.cam.LUTEnable.SetValue(False)

                print('setting frame rate enable to False')
                self.cam.AcquisitionFrameRateEnable.SetValue(False)
                print('setting TriggerMode On')
                self.cam.TriggerMode.SetIntValue(PySpin.TriggerMode_Off)


            elif settings == 3:
                print('Acquisition and Trigger Settings: 3 (software)')
                print('setting continuous acquisition mode')
                self.cam.AcquisitionMode.SetIntValue(PySpin.AcquisitionMode_Continuous)

                print('setting Look Up Table Enable to False')
                self.cam.LUTEnable.SetValue(False)

                print('setting frame rate enable to True')
                self.cam.AcquisitionFrameRateEnable.SetValue(True)
                print('setting frame rate to 1')
                self.cam.AcquisitionFrameRate.SetValue(20.0)
                print('setting TriggerMode Off')
                self.cam.TriggerMode.SetIntValue(PySpin.TriggerMode_Off)
                print('setting TriggerSource Line0')
                self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
                print('setting TriggerSource Software')
                self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)



    def recording_init(self, filename = None, N_frames = 1200, comments = ''):
        from os.path import exists
        from h5py import File
        from time import time, sleep, ctime
        from numpy import zeros
        if filename is None:
            filename = self.recording_root+f'{self.name}_{comments}.raw.hdf5'
        if exists(filename):
            print('----WARNING----')
            print('The HDF5 file exists. Please delete it first!')
        else:
            print(f'The HDF5 was created. The file name is {filename}')
            with File(filename,'a') as f:
                f.create_dataset('exposure time', data = 'trigger width')
                f.create_dataset('black level all', data = self.black_level['all'])
                f.create_dataset('black level analog', data = self.black_level['analog'])
                f.create_dataset('black level digital', data = self.black_level['digital'])
                f.create_dataset('gain', data = self.gain)
                f.create_dataset('time', data = ctime(time()))
                f.create_dataset('temperature', data = self.temperature)
                f.create_dataset('images', (N_frames,self.height,self.width), dtype = 'int16', chunks = (1,self.height,self.width))
                f.create_dataset('timestamps_lab', (N_frames,) , dtype = 'float64')
                f.create_dataset('timestamps_camera', (N_frames,) , dtype = 'float64')
            self.recording_filename = filename
        self.recording_N_frames = N_frames
        self.recording_pointer = 0
        self.recording_buffer = zeros((self.recording_N,self.height,self.width))

    def record_once(self,filename,N):
        from h5py import File
        from numpy import copy, array

        self.recording_buffer = self.queue.dequeue(N)
        images = self.recording_buffer

        if images.shape[0] > 0:
            with File(filename,'a') as f:
                for i in range(N):
                    pointer = self.recording_pointer
                    image = images[i]
                    tlab,tcam = self.extract_timestamp_image(image)
                    f['images'][pointer] = image[:self.height,:]
                    f['timestamps_lab'][pointer] = tlab
                    f['timestamps_camera'][pointer] = tcam
                    self.recording_pointer += 1

        else:
            print(f'{self.name}: got empty array from deque with rear value in the queue of {self.queue.rear}')

    def recording_run(self):
        from h5py import File
        from time import time,ctime
        filename = self.recording_filename
        while (self.recording) and (self.recording_pointer<self.recording_N_frames):
            N = self.recording_N
            if self.queue.length > N:
                self.record_once(filename = filename,N = N)
            else:
                sleep((N+3)*0.051)
        self.recording = False
        print(ctime(time()),f'Recording of {self.recording_N_frames} is Finished')

    def recording_chunk_run(self,M):
        from h5py import File
        filename = self.recording_filename
        i = 0
        while (i < M) and self.recording:
            N = self.recording_N
            if self.queue.length > N:
                self.record_once(filename = filename,N = N)
                i+=N
            else:
                sleep((N+3)*self._exposure_time/1000000)
        self.recording = False
        print(f'{self.recording_filename}: recording stopped')

    def recording_start(self):
        from ubcs_auxiliary.multithreading import new_thread
        self.recording = True
        new_thread(self.recording_run)

    def recording_stop(self):
        self.recording = False

    def recording_scratch(self):
        root = self.recording_root + 'scratch/'
        def full_speed():
            filename = self.recording_filename
            if images.shape[0] > 0:
                for i in range(N):
                    if self.recording_pointer>=self.recording_N_frames:
                        self.recording = False
                        break
                        print('Recording of {self.recording_N_frames} is Finished')
                    pointer = self.recording_pointer
                    image = images[i]
                    with File(filename,'a') as f:
                        f['images'][pointer] = image[:self.height,:]
                        tlab,tcam = self.extract_timestamp_image(image)
                        f['timestamps_lab'][pointer] = tlab
                        f['timestamps_camera'][pointer] = tcam
                        #f['frameID'][pointer] = tcam
                        self.recording_pointer += 1

    def extract_timestamp_image(self,arr):
        """
        Takes array of images (length, rows, cols) and returns the header data

        The header information is located in the last row (row = 3000) of an image
        """
        header_arr = arr[-1,:64]
        timestamp_lab =  self.binarr_to_number(header_arr)/1000000
        header_arr = arr[-1,64:128]
        timestamp_cam =  self.binarr_to_number(header_arr)
        return timestamp_lab,timestamp_cam

    def binarr_to_number(self,vector):
        num = 0
        from numpy import flip
        vector = flip(vector)
        length = vector.shape[0]
        for i in range(length):
           num += (2**(i))*vector[i]
        return num

    def bin_array(self,num, m):
        from numpy import uint8, binary_repr,array
        """Convert a positive integer num into an m-bit bit vector"""
        return array(list(binary_repr(num).zfill(m))).astype(uint8)

def get_gain_mean_std(camera):
    from numpy import array
    from time import sleep
    lst = []
    for g in range(0,49):
        camera.gain = g
        gain = camera.gain
        camera.trigger_now();
        sleep(0.3)
        img = camera.buffer.get_last_value()[0]
        print(gain,img.mean(),img.std())
        lll = [gain,img.mean(),img.std()]
        lst.append(lll)
    return array(lst)

if __name__ == '__main__':
    from PySpin import System
    system = System.GetInstance()
    camera = FlirCamera('test',system)
    camera.get_all_cameras()

    print("camera_dm4 = FlirCamera('dm4', system)")
    print("camera_dm4.init(serial_number = '19490369')")
    print('camera_dm4.start_thread()')

    print("camera_dm34 = FlirCamera('dm34',system)")
    print("camera_dm34.init(serial_number = '18159480')")
    print('camera_dm34.start_thread()')

    print("camera_dm16 = FlirCamera('dm16',system)")
    print("camera_dm16.init(serial_number = '18159488')")
    print('camera_dm16.start_thread()')

    print("camera_out = FlirCamera('dmout',system)")
    print("camera_out.init(serial_number = '20130136')")
    print('camera_out.start_thread()')
    from matplotlib import pyplot as plt
    plt.ion()
