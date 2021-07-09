from time import time, sleep
from numpy import zeros, right_shift, array, copy
import PySpin
import logging
from logging import debug, info, warning, error
logging.getLogger(__name__).addHandler(logging.NullHandler())

from PySpin import PixelFormat_Mono16,PixelFormat_Mono12Packed
import sys
import os

class FlirCamera():
    def __init__(self, name = 'temp', system = None):
        """
        """
        debug(f'__init__ FlirCamera with name {name}')
        from numpy import zeros, nan
        self.name = name
        self.system = system

        #Acquisition
        self.queue_length = 32
        self.acquiring = False
        self.header_length = 4096

        #Recording
        self.recording_filename = f'camera_{name}.hdf5'
        self.recording_root = '/mnt/data/'
        self.recording_N = 1
        self.recording = False
        self.recording_pointer = 0
        self.recording_chunk_pointer = 0
        self.recording_chunk_maxpointer = 65535

        self.write_to_hdf5_benchmark = []

        self.threads = {}

        #configuration Parameters
        self.nice = 0
        self.reverseX = 0
        self.reverseY = 0
        self.rotate = 0
        self.trigger = 'Software'

        self.binning_selector = 'All'
        self.binning_horizontal_mode = 'Sum'
        self.binning_vertical_mode = 'Sum'
        self.binning_vertical = int(1)
        self.binning_horizontal = int(1)

        self.calc_on_the_fly = False
        self.broadcast_frames = False

        self.io_pull_queue = None
        self.io_push_queue = None

        self.system = PySpin.System.GetInstance()

    def init(self, config_filename):
        from numpy import zeros, nan, ones
        info('')
        info('-------------INITIALIZING NEW CAMERA-----------')
        import PySpin
        from circular_buffer_numpy.queue import Queue

        config, flag = self.read_config_file(config_filename)
        self.config = config

        self.name = config['name']

        self.pixel_format = config['pixel_format']

        self.ROI_width = int(config['ROI_width'])
        self.ROI_height = int(config['ROI_height'])
        self.ROI_offset_x = int(config['ROI_offset_x'])
        self.ROI_offset_y = int(config['ROI_offset_y'])

        self.reverseX = int(config['ROI_offset_y'])
        self.reverseY = int(config['ROI_offset_y'])
        self.rotate = int(config['rotate'])
        self.trigger = config['trigger']
        if 'binning_selector' in config.keys():
            self.binning_selector = config['binning_selector']
        if 'binning_horizontal_mode' in config.keys():
            self.binning_horizontal_mode = config['binning_horizontal_mode']
        if 'binning_vertical_mode' in config.keys():
            self.binning_vertical_mode = config['binning_vertical_mode']
        if 'binning_vertical' in config.keys():
            self.binning_vertical = int(config['binning_vertical'])
        if 'binning_horizontal' in config.keys():
            self.binning_horizontal = int(config['binning_horizontal'])

        nice = int(config['nice'])
        self.nice = nice
        info(f'setting up nice {nice}')
        if nice != 0:
            os.nice(nice)
        sn = str(config['serial_number'])

        self.cam = self.find_camera(serial_number = sn)

        if self.cam is not None:
            self.cam.Init()

        for i in range(3):
            try:
                self.cam.BeginAcquisition()
                info("Acquisition Started")
                sleep(1)
            except PySpin.SpinnakerException as ex:
                info("Acquisition was already started")
                sleep(1)
            try:
                self.cam.EndAcquisition()
                info("Acquisition ended")
                sleep(1)
            except PySpin.SpinnakerException as ex:
                info("Acquisition was already ended")
                sleep(1)

        # reset camera to default settings

        self.nodenames = self.get_nodenames()

        #Transport Configuration
        self.configure_transport()

        #Analog Configuration
        self.conf_acq_and_trigger()

        #Configure Image Format
        self.configure_image()

        self.height = self.get_height()
        self.width = self.get_width()
        if (self.pixel_format == 'mono8'):
            from lcp_video.analysis import get_mono12p_conversion_mask,get_mono12p_conversion_mask_8bit
            self.img_len = int(self.height*self.width)
            self.images_dtype = 'uint8'
            self.conversion_mask = get_mono12p_conversion_mask(self.img_len)
            self.conversion_mask8 = get_mono12p_conversion_mask_8bit(self.img_len)
        elif (self.pixel_format == 'mono12p'):
            from lcp_video.analysis import get_mono12p_conversion_mask,get_mono12p_conversion_mask_8bit
            self.img_len = int(self.height*self.width*1.5)
            self.images_dtype = 'uint8'
            self.conversion_mask = get_mono12p_conversion_mask(self.img_len)
            self.conversion_mask8 = get_mono12p_conversion_mask_8bit(self.img_len)
        elif (self.pixel_format == 'mono12packed'):
            from lcp_video.analysis import get_mono12packed_conversion_mask,get_mono12p_conversion_mask_8bit
            self.img_len = int(self.height*self.width*1.5)
            self.images_dtype = 'uint8'
            self.conversion_mask = get_mono12packed_conversion_mask(self.img_len)
            self.conversion_mask8 = get_mono12p_conversion_mask_8bit(self.img_len)
        elif self.pixel_format == 'mono16' or self.pixel_format == 'mono12p_16':
            self.img_len = int(self.height*self.width)
            self.images_dtype = 'int16'
        self.queue = Queue((self.queue_length,self.img_len+self.header_length), dtype = self.images_dtype)

        self.queue_frameID = Queue((self.queue_length,2), dtype = 'float64')

        self.last_raw_image = zeros((self.img_len+self.header_length,), dtype = self.images_dtype)

        from circular_buffer_numpy.circular_buffer import CircularBuffer
        self.hits_buffer = CircularBuffer((1350000,2),dtype = 'float64')



        #Algorithms Configuration
        if 'lut_enable' in config.keys():
            self.lut_enable = config['lut_enable']
        else:
            self.lut_enable = False
        if 'gamma_enable' in config.keys():
            self.gamma_enable = config['gamma_enable']
        else:
            self.gamma_enable = False




        try:
            self.cam.AcquisitionStop()
            info("Acquisition stopped")
        except PySpin.SpinnakerException as ex:
            info("Acquisition was already stopped")

        self.image_threshold = zeros((self.height,self.width))+7

        self.image_median = zeros((self.height,self.width))
        self.image_median[:,:]= 15

        self.image_mean = zeros((self.height,self.width))
        self.image_mean[:,:]= 15
        self.image_mean_flag = True

        self.image_std = ones((self.height,self.width))
        self.image_std[:,:] = 0.8
        self.image_std_flag = True

        self.sigma_level = 6

        self.mask = zeros((self.height,self.width),dtype ='bool')
        self.mask_flag = True

        self.last_frameID = -1
        self.num_of_missed_frames = 0

        self.start_thread()


    def read_config_file(self, filename):
        import yaml
        import os
        flag =  os.path.isfile(filename)
        info(f'the file {filename} exists {flag}. Reading config file...')
        if flag:
            with open(filename,'r') as handle:
                config = yaml.safe_load(handle.read())  # (2)
        else:
            config = {}
        return config, flag

    def check_node(self,node):
        from PySpin import IsReadable, IsWritable, IsAvailable
        res = {}
        res['IsAvailable'] = f'{IsAvailable(node)}:<5'
        res['IsReadable'] = f'{IsReadable(node)}:<5'
        res['IsWritable'] = f'{IsWritable(node)}:<5'
        if res['IsReadable']:
            res['Value'] = f'{node.GetValue()}:<5'
            try:
                res['Symbolic'] = f'{node.GetCurrentEntry().GetSymbolic()}:<15'
            except:
                res['Symbolic'] = f'{None}:<15'
        return res

    def check_used_nodes(self):
        #check if the camera FLIR API is initialized. If not, initialize it and remember the original state
        if self.cam.IsInitialized():
            flag = True
            pass
        else:
            flag = False
            self.cam.Init()
        nodedict = self.get_nodedict()
        for nodekey in list(nodedict.keys()):
            node = nodedict[nodekey]
            if node is not None:
                print(f'{node.GetName():<15},{self.check_node(node)}')
            else:
                print("'{nodename} doesn't exist")

        #Revert to original state
        if not flag:
            self.cam.DeInit()


    def read_current_setting(self):
        """
        """
        from PySpin import IsReadable

        exposure_time = self.cam.ExposureTime.GetValue()
        print(f'exposure time, us = {round(exposure_time,0)}')

        # Timed TriggerWidth
        exposure_mode = self.cam.ExposureMode.GetCurrentEntry().GetSymbolic()
        print(f'exposure mode = {exposure_mode}')

        trigger_source = self.cam.TriggerSource.GetCurrentEntry().GetSymbolic()
        print(f'trigger source = {trigger_source}')

        if PySpin.IsReadable(self.cam.TriggerSelector):
            trigger_selector = self.cam.TriggerSelector.GetCurrentEntry().GetSymbolic()
            print(f'trigger selector = {trigger_selector}')
        else:
            print(f'trigger selector is not readable')

        if PySpin.IsReadable(self.cam.TriggerActivation):
            trigger_activation = self.cam.TriggerActivation.GetCurrentEntry().GetSymbolic()
            print(f'trigger activation = {trigger_activation}')
        else:
            print(f'trigger activation is not readable')

        acquisition_frame_rate_enable = self.cam.AcquisitionFrameRateEnable.GetValue()
        print(f'acquisition frame rate enable = {acquisition_frame_rate_enable}')

        acquisition_frame_rate = self.cam.AcquisitionFrameRate.GetValue()
        print(f'acquisition frame rate = {acquisition_frame_rate}')

        x_offset = self.cam.OffsetX.GetValue()
        y_offset = self.cam.OffsetY.GetValue()
        print(f'Offset (x,y) = ({x_offset},{y_offset})')

        width = self.cam.Width.GetValue()
        height = self.cam.Height.GetValue()
        print(f'frame (width,height) = ({width},{height})')


        pixel_format  = self.cam.PixelFormat.GetCurrentEntry().GetSymbolic()
        print(f'pixel format = {pixel_format}')

        binning_selector  = self.cam.BinningSelector.GetCurrentEntry().GetSymbolic()
        print(f'binning selector = {binning_selector}')

        binning_horizontal_mode  = self.cam.BinningHorizontalMode.GetCurrentEntry().GetSymbolic()
        print(f'binning horizontal mode = {binning_horizontal_mode}')

        binning_vertical_mode  = self.cam.BinningVerticalMode.GetCurrentEntry().GetSymbolic()
        print(f'binning vertical mode = {binning_vertical_mode}')

        binning_horizontal  = self.cam.BinningHorizontal.GetValue()
        print(f'binning horizontal = {binning_horizontal}')

        binning_vertical = self.cam.BinningVertical.GetValue()
        print(f'binning vertical = {binning_vertical}')


        dt = 3
        p1 = self.queue.global_rear; sleep(dt); p2 = self.queue.global_rear; print(f'experimentaly measure acquisition rate in {dt} seconds = {round((p2-p1)/dt,2) }')


    def close(sself):
        pass

    def kill(self):
        """
        """
        self.pause_acquisition()
        sleep(1)
        self.stop_acquisition()
        sleep(1)
        del self.cam
        self.system.ReleaseInstance()
        del self

    def reset_to_factory_settings(self):
        #check of camera is initalized. It not, initialize the camera. variable flag is used to know the state of INIT at the beginning of this function
        if self.cam.IsInitialized():
            flag = False
        else:
            self.cam.Init()
            flag = True
        self.pause_acquisition()
        self.stop_acquisition()
        try:
            self.cam.AcquisitionStop()
            print ("Acquisition stopped")
        except PySpin.SpinnakerException as ex:
            print("Acquisition was already stopped")
        self.cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
        self.cam.UserSetLoad()

        # Check teh original status of INIT and return to the original state.
        if not flag:
            self.cam.DeInit()

    def configure_transport(self):
        cam = self.cam
        # Configure Transport Layer Properties
        cam.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_OldestFirst)
        cam.TLStream.StreamBufferCountMode.SetValue(PySpin.StreamBufferCountMode_Manual )
        cam.TLStream.StreamBufferCountManual.SetValue(20)
        info(f"Buffer Handling Mode: {cam.TLStream.StreamBufferHandlingMode.GetCurrentEntry().GetSymbolic()}")
        info( f"Buffer Count Mode: {cam.TLStream.StreamBufferCountMode.GetCurrentEntry().GetSymbolic()}")
        info(f"Buffer Count: {cam.TLStream.StreamBufferCountManual.GetValue()}")
        info(f"Max Buffer Count: {cam.TLStream.StreamBufferCountManual.GetMax()}")

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
        info(f'found {num_cameras} cameras')
        for i,cam in enumerate(cam_list):
            sn = cam.TLDevice.DeviceSerialNumber.GetValue()
            print(f'sn = {sn}')
            if serial_number == sn:
                self.serial_number = sn
                break
            else:
                cam = None
                self.serial_number = None
        cam_list.Clear()
        return cam

    def get_all_cameras(self):
        cam_list = self.system.GetCameras()
        num_cameras = cam_list.GetSize()
        info(f'found {num_cameras} cameras')
        cameras = []
        for i,cam in enumerate(cam_list):
            sn = cam.TLDevice.DeviceSerialNumber.GetValue()
            info(f'sn = {sn}')
            cameras.append(sn)
        cam_list.Clear()
        return cameras

    def get_nodenames(self):
        # import PySpin
        # self.cam.Init()
        # # Retrieve GenICam nodemap
        # nodemap = self.cam.GetNodeMap()
        # self.nodemap = nodemap
        nodes = []
        nodes.append('Gain')
        nodes.append('GainAuto')

        nodes.append('ExposureAuto')
        nodes.append('ExposureTime')
        nodes.append('ExposureMode')

        nodes.append('BlackLevelSelector')
        nodes.append('BlackLevel')
        nodes.append('BlackLevelAuto')

        nodes.append('TriggerSource')
        nodes.append('TriggerSelector')
        nodes.append('TriggerActivation')
        nodes.append('TriggerMode')
        nodes.append('TriggerOverlap')

        nodes.append('AcquisitionMode')
        nodes.append('AcquisitionFrameRateEnable')
        nodes.append('AcquisitionFrameRate')

        nodes.append('Height')
        nodes.append('Width')
        nodes.append('OffsetX')
        nodes.append('OffsetY')
        nodes.append('PixelFormat')
        nodes.append('ReverseX')
        nodes.append('ReverseY')


        nodes.append('LUTEnable')
        nodes.append('IspEnable')
        nodes.append('BinningSelector')
        nodes.append('BinningHorizontalMode')
        nodes.append('BinningVerticalMode')
        nodes.append('BinningHorizontal')
        nodes.append('BinningVertical')
        return nodes

    def get_nodedict(self):
        nodes = {}
        nodes['Gain'] = self.cam.Gain
        nodes['GainAuto'] = self.cam.GainAuto

        nodes['ExposureTime'] = self.cam.ExposureTime
        nodes['ExposureMode'] = self.cam.ExposureMode
        nodes['ExposureAuto'] = self.cam.ExposureAuto

        nodes['BlackLevelSelector'] = self.cam.BlackLevelSelector
        nodes['BlackLevel'] = self.cam.BlackLevel
        #nodes['BlackLevelAuto'] = self.cam.BlackLevelAuto

        nodes['TriggerSource'] = self.cam.TriggerSource
        nodes['TriggerSelector'] = self.cam.TriggerSelector
        nodes['TriggerActivation'] = self.cam.TriggerActivation
        nodes['TriggerMode'] = self.cam.TriggerMode
        nodes['TriggerOverlap'] = self.cam.TriggerOverlap

        nodes['AcquisitionMode'] = self.cam.AcquisitionMode
        nodes['AcquisitionFrameRateEnable'] = self.cam.AcquisitionFrameRateEnable
        nodes['AcquisitionFrameRate'] = self.cam.AcquisitionFrameRate

        nodes['Height'] = self.cam.Height
        nodes['Width'] = self.cam.Width
        nodes['OffsetX'] = self.cam.OffsetX
        nodes['OffsetY'] = self.cam.OffsetY
        nodes['PixelFormat'] = self.cam.PixelFormat
        nodes['ReverseX'] = self.cam.ReverseX
        nodes['ReverseY'] = self.cam.ReverseY


        nodes['LUTEnable'] = self.cam.LUTEnable
        nodes['IspEnable'] = self.cam.IspEnable
        nodes['BinningSelector'] = self.cam.BinningSelector
        nodes['BinningHorizontalMode'] = self.cam.BinningHorizontalMode
        nodes['BinningVerticalMode'] = self.cam.BinningVerticalMode
        nodes['BinningHorizontal'] = self.cam.BinningHorizontal
        nodes['BinningVertical']  = self.cam.BinningVertical
        return nodes

    def get_image_background(self,N=0):
        """
        function that acquires N images and calculates M1, M2 and threshold images.
        """
        from time import ctime, time, sleep
        import os
        from numpy import sqrt

        from lcp_video.procedures.procedures import stats_from_chunk,find_pathnames
        # Load or compute stats for first chunk in dataset.
        info(f'starting acquiring background image file')
        self.recording_stop();
        old_recording_chunk_maxpointer = self.recording_chunk_maxpointer
        self.recording_chunk_maxpointer = 2
        sleep(1);
        string = ctime().replace(' ','-').replace(':','-')
        self.recording_init(N_frames = 256, name = f'background-{string}', overwrite = True);
        self.queue.reset();
        self.recording_start()
        filename = self.recording_basefilename+'_0.raw.hdf5'
        while not os.path.exists(filename):
            sleep(1)
        self.recording_stop()
        os.remove(self.recording_basefilename+'_1.tmpraw.hdf5')
        self.recording_chunk_maxpointer = old_recording_chunk_maxpointer
        # analysing raw file and creating .stats. and returning
        root = self.recording_root
        terms = [self.recording_basefilename,'.raw.hdf5']
        median,mean,var,threshold = stats_from_chunk(find_pathnames(root,terms)[0])
        self.image_mean = mean
        self.image_std = sqrt(var)
        self.image_var = var
        self.image_median = median
        self.image_threshold = threshold

        info(f'starting acquiring background image file')

    def get_image_background_old(self,N=0):
        """
        function that acquires N images and calculates M1, M2 and threshold images.
        """
        from numpy import mean, var, copy, zeros,std, sqrt
        if N == 0:
            N = self.queue.length
        raw  = copy(self.queue.peek_last_N(N))
        img = zeros((N,self.height,self.width))
        for i in range(N):
            img[i] = self.convert_raw_to_image(raw[i])
        self.image_mean = mean(img, axis = 0)
        self.image_std = sqrt(var(img, axis = 0) + 0.5)

    def get_image(self):
        from lcp_video.analysis import mono12p_to_image
        self.last_raw_image *= 0
        if self.acquiring:
            image_result = self.cam.GetNextImage()
            timestamp = image_result.GetTimeStamp()
            frameid = image_result.GetFrameID()
            info(f'get : {timestamp},    {frameid}')
            if (self.last_frameID != -1) and ((frameid-self.last_frameID) != 1):
                missed = frameid-self.last_frameID
                self.num_of_missed_frames += missed
                info(f'missed {missed} frames. {self.queue.global_rear}. Current frame ID {frameid}, last frame ID {self.last_frameID} ')
            self.last_frameID = frameid
            # Getting the image data as a numpy array
            image_res = image_result.GetData()
            if self.pixel_format == 'mono12p_16':
                image_data = mono12p_to_image(image_res, self.height, self.width)
            else:
                image_data = image_res
            image_result.Release()
        else:
            info('No Data in get image')
            image_data = zeros((self.height*self.width,))

        pointer = self.img_len
        self.last_raw_image[:pointer] = image_data
        self.last_raw_image[pointer:pointer+64] = self.get_image_header(value =int(time()*1000000), length = 64)
        self.last_raw_image[pointer+64:pointer+128] = self.get_image_header(value = timestamp, length = 64)
        self.last_raw_image[pointer+128:pointer+192] = self.get_image_header(value = frameid, length = 64)
        return self.last_raw_image

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

    def bin_array(num, m):
        from numpy import uint8, binary_repr,array
        """Convert a positive integer num into an m-bit bit vector"""
        return array(list(binary_repr(num).zfill(m))).astype(uint8)

    def run_once(self):
        from time import time
        from numpy import zeros,array
        from lcp_video.analysis import mono12p_to_image
        if self.acquiring:
            raw = self.get_image().reshape(1,self.img_len+4096)
            self.queue.enqueue(raw)
            if not self.recording and self.calc_on_the_fly:
                self.last_reshaped_image = mono12p_to_image(raw[0,:self.img_len],self.height,self.width).reshape((self.height,self.width))
                hits = ((self.last_reshaped_image>(self.image_threshold+self.image_median))*~self.mask).sum()
                arr = zeros((1,2))
                arr[0,0] = time()
                arr[0,1] = hits
                from EPICS_CA.CAServer import casput
                casput(f'{self.name.upper()}_CAMERA:HITS.RBV',hits)
                self.hits_buffer.append(arr)
            if self.broadcast_frames:
                io_dict = {}
                tlab,tcam,frameID = self.extract_timestamp_image(raw.flatten())
                io_dict['image'] = raw[:,:self.img_len].flatten()
                io_dict['timestamp_lab'] = tlab
                io_dict['timestamp_camera'] = tcam
                io_dict['frameID'] = frameID
                self.io_put(io_dict)


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
            info("Acquisition was already ended")

    def io_put(self, io_dict):
        if self.io_push_queue is not None:
            self.io_push_queue.put(io_dict)

    def io_get(self):
        pass

    def get_black_level(self):
        import traceback
        from numpy import nan
        all = nan
        analog = nan
        digital = nan

        try:
            self.cam.BlackLevelSelector.SetValue(0)
        except:
            error(f'The self.cam.BlackLevelSelector.SetValue(0) failed {traceback.format_exc()}')
        try:
            all = self.cam.BlackLevel.GetValue()*4095/100
        except:
            error(f'self.cam.BlackLevel.GetValue()*4095/100 failed {traceback.format_exc()}')
        try:
            self.cam.BlackLevelSelector.SetValue(1)
            analog = self.cam.BlackLevel.GetValue()*4095/100
        except:
            error(f'sself.cam.BlackLevelSelector.SetValue(1); analog = self.cam.BlackLevel.GetValue()*4095/100 failed {traceback.format_exc()}')
        try:
            self.cam.BlackLevelSelector.SetValue(2)
            digital = self.cam.BlackLevel.GetValue()*4095/100
        except:
            error(f'self.cam.BlackLevelSelector.SetValue(2); digital = self.cam.BlackLevel.GetValue()*4095/100 failed {traceback.format_exc()}')
        try:
            self.cam.BlackLevelSelector.SetValue(0)
        except:
            error(f'self.cam.BlackLevelSelector.SetValue(0) failed {traceback.format_exc()}')
        return {'all':all,'analog':analog,'digital':digital}

    def set_black_level(self,value):
        """
        """
        try:
            self.cam.BlackLevelSelector.SetValue(0)
            self.cam.BlackLevel.SetValue(value*100/4095)
        except:
            error(f'self.cam.BlackLevelSelector.SetValue(0); self.cam.BlackLevel.SetValue(value*100/4095) failed {traceback.format_exc()}')
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
        """
        returns Symbolic value of GainAuto
        """
        if self.cam is not None:
            return self.cam.GainAuto.GetCurrentEntry().GetSymbolic()
        else:
            return None

    def set_autogain(self,value):
        from PySpin import GainAuto_Off,GainAuto_Once,GainAuto_Continuous
        node = self.cam.GainAuto
        if value == 'Off':
            node.SetValue(GainAuto_Off)
        elif value == 'Once':
            node.SetValue(GainAuto_Once)
        elif value == 'Continuous':
            node.SetValue(GainAuto_Continuous)
    autogain = property(get_autogain,set_autogain)

    def get_serial_number(self):
        import PySpin
        device_serial_number = self.cam.TLDevice.DeviceSerialNumber.GetValue()
        return device_serial_number

    def get_acquisition_mode(self):
        node = self.cam.AcquisitionMode
        return node.GetValue()

    def set_acquisition_mode(self,value):
        node = self.cam.AcquisitionMode
        continious = node.GetEntryByName("Continuous")
        single_frame = node.GetEntryByName("SingleFrame")
        multi_frame = node.GetEntryByName("MultiFrame")
        if value == 'Continuous':
            mode = continious
        elif value == 'SingleFrame':
            mode = single_frame
        elif value == 'MultiFrame':
            mode = multi_frame
        info(f'setting acquisition mode {value}')
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
            info('setting Look Up Table Enable to False')
            self.cam.LUTEnable.SetValue(value)
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
        info(f'setting gamma enable {value}')
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
        info(f'setting gamma {value}')
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
        info(f'Setting up ExposureMode to {mode}')
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
            info(f'setting gamma enable {value}')
    def get_autoexposure(self):
        value = self.cam.ExposureAuto.GetValue()
        if value == 0:
            return 'Off'
        elif value == 1:
            return 'Once'
        elif value == 2:
            return 'Continuous'
        else:
            return 'unknown'
    autoexposure = property(get_autoexposure,set_autoexposure)

    def set_exposure_time(self, value):
        if self.cam is not None:
            self.cam.ExposureTime.SetValue(value)
            self._exposure_time = value
        else:
            pass
    def get_exposure_time(self,):
        if self.cam is not None:
            try:
                return self.cam.ExposureTime.GetValue()
            except:
                return -1
        else:
            return nan
        self._exposure_time = value
    exposure_time = property(get_exposure_time,set_exposure_time)

    def trigger_now(self):
        """
        software trigger for camera
        """
        self.cam.TriggerSoftware()

    def configure_image(self):
        import PySpin

        "Two different pixel formats: PixelFormat_Mono12p and PixelFormat_Mono12Packed"

        if self.reverseX == 1:
            print('reversing X axis')
            self.cam.ReverseX.SetValue(True)#(True,True)
        else:
            self.cam.ReverseX.SetValue(False)#(False,True)
        if self.reverseY == 1:
            print('reversing Y axis')
            self.cam.ReverseY.SetValue(True)#(True,True)
        else:
            self.cam.ReverseY.SetValue(False)#(False,True)

        info(f'setting pixel format {self.pixel_format}')
        if self.pixel_format=='mono12p' or self.pixel_format=='mono12p_16':
            try:
                self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono12p)
            except:
                print('cannot set PixelFormat.SetValue(PySpin.PixelFormat_Mono12p)')

        elif self.pixel_format =='mono12packed':
            try:
                self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono12Packed)
            except:
                print('cannot set PixelFormat.SetValue(PySpin.PixelFormat_Mono12Packed)')
        elif self.pixel_format =='mono16':
            try:
                self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
            except:
                print('cannot set PixelFormat.SetValue(PySpin.PixelFormat_Mono16)')
        elif self.pixel_format =='mono8':
            try:
                self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
            except:
                print('cannot set PixelFormat.SetValue(PySpin.PixelFormat_Mono8)')

        #self.cam.Width.SetValue(self.cam.WidthMax.GetValue())
        #self.cam.Height.SetValue(self.cam.HeightMax.GetValue())

        width_max = self.cam.WidthMax.GetValue()
        height_max = self.cam.HeightMax.GetValue()

        old_offset_x = self.cam.OffsetX.GetValue()
        old_offset_y = self.cam.OffsetY.GetValue()
        new_offset_x = self.ROI_offset_x
        new_offset_y = self.ROI_offset_y

        old_width = self.cam.Width.GetValue()
        old_height = self.cam.Height.GetValue()
        new_height = self.ROI_height
        new_width = self.ROI_width
        try:
            if new_offset_x <= old_offset_x:
                info(f'setting ROI_offset_x: {self.ROI_offset_x}')
                self.cam.OffsetX.SetValue(self.ROI_offset_x)
                info(f'setting ROI_width: {self.ROI_width}')
                self.cam.Width.SetValue(self.ROI_width)
            else:
                info(f'setting ROI_width: {self.ROI_width}')
                self.cam.Width.SetValue(self.ROI_width)
                info(f'setting ROI_offset_x: {self.ROI_offset_x}')
                self.cam.OffsetX.SetValue(self.ROI_offset_x)
            if new_offset_y <= old_offset_y:
                info(f'setting ROI_offset_x: {self.ROI_offset_y}')
                self.cam.OffsetY.SetValue(self.ROI_offset_y)
                info(f'setting ROI_width: {self.ROI_width}')
                self.cam.Height.SetValue(self.ROI_height)
            else:
                info(f'setting ROI_width: {self.ROI_width}')
                self.cam.Width.SetValue(self.ROI_width)
                info(f'setting ROI_offset_x: {self.ROI_offset_x}')
                self.cam.OffsetX.SetValue(self.ROI_offset_x)

            info(f'setting ROI_offset_x: {self.ROI_offset_x}')
            self.cam.OffsetX.SetValue(self.ROI_offset_x)
            info(f'setting ROI_width: {self.ROI_width}')
            self.cam.Width.SetValue(self.ROI_width)

            info(f'setting ROI_height: {self.ROI_height}')
            self.cam.Height.SetValue(self.ROI_height)
            info(f'setting ROI_offset_y: {self.ROI_offset_y}')
            self.cam.OffsetY.SetValue(self.ROI_offset_y)
        except:
            pass

        info('setting up binning')
        self.cam.BinningHorizontal.SetValue(1)
        self.cam.BinningVertical.SetValue(1)
        self.cam.IspEnable.SetValue(False)




        if self.binning_horizontal_mode == 'Average':
            self.cam.BinningHorizontalMode.SetValue(PySpin.BinningHorizontalMode_Average)
        elif self.binning_horizontal_mode == 'Sum':
            self.cam.BinningHorizontalMode.SetValue(PySpin.BinningHorizontalMode_Sum)

        if self.binning_vertical_mode == 'Average':
            self.cam.BinningVerticalMode.SetValue(PySpin.BinningVerticalMode_Average)
        elif self.binning_vertical_mode == 'Sum':
            self.cam.BinningVerticalMode.SetValue(PySpin.BinningVerticalMode_Sum)


        self.cam.BinningHorizontal.SetValue(self.binning_horizontal)
        self.cam.BinningVertical.SetValue(self.binning_vertical)

        if self.binning_selector == 'All':
            self.cam.BinningSelector.SetValue(PySpin.BinningSelector_All)
        elif self.binning_selector == 'Sensor':
            self.cam.BinningSelector.SetValue(PySpin.BinningSelector_Sensor)
        elif self.binning_selector == 'ISP':
            self.cam.BinningSelector.SetValue(PySpin.BinningSelector_ISP)

    def conf_acq_and_trigger(self, settings = 1):
        """
        a collection of setting defined as appropriate for the selected modes of operation.

        settings 1 designed for external trigger mode of operation
        Acquisition Mode = Continuous
        Acquisition FrameRate Enable = False
        TriggerSource = Line 0
        TriggerMode = On
        TriggerSelector = FrameStart
        """


        if self.cam is not None:

            self.autoexposure = self.config['autoexposure']
            self.autogain = self.config['autogain']
            self.set_exposure_mode('Timed')
            self.exposure_time = float(self.config['exposure_time'])
            self.gain = float(self.config['gain'])
            self.black_level = float(self.config['black_level'])

            if settings ==1:
                info('Acquisition and Trigger Settings: 1')
                info('setting continuous acquisition mode')
                self.cam.AcquisitionMode.SetIntValue(PySpin.AcquisitionMode_Continuous)
                info('setting frame rate enable to False')
                self.cam.AcquisitionFrameRateEnable.SetValue(False)

                if self.trigger == 'Line0':
                    info('setting TriggerSource Line0')
                    self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
                    info('setting TriggerMode On')
                    self.cam.TriggerMode.SetIntValue(PySpin.TriggerMode_On)
                    info('setting TriggerSelector FrameStart')
                    self.cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
                    info('setting TriggerActivation RisingEdge')
                    self.cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
                    info('setting TriggerOverlap ReadOnly ')
                    self.cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
                elif self.trigger == "Software":
                    info('setting TriggerSource Software')
                    self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
                    info('setting TriggerMode On')
                    self.cam.TriggerMode.SetIntValue(PySpin.TriggerMode_Off)
                else:
                    info('setting TriggerSource Software')
                    self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
                    info('setting TriggerMode On')
                    self.cam.TriggerMode.SetIntValue(PySpin.TriggerMode_Off)





            elif settings ==2:
                info('Acquisition and Trigger Settings: 2')
                info('setting SingleFrame acquisition mode')
                self.cam.AcquisitionMode.SetIntValue(PySpin.AcquisitionMode_SingleFrame)
                info('setting frame rate enable to False')
                self.cam.AcquisitionFrameRateEnable.SetValue(False)

                info('setting frame rate enable to False')
                self.cam.AcquisitionFrameRateEnable.SetValue(False)
                info('setting TriggerMode On')
                self.cam.TriggerMode.SetIntValue(PySpin.TriggerMode_Off)


            elif settings == 3:
                info('Acquisition and Trigger Settings: 3 (software)')
                info('setting continuous acquisition mode')
                self.cam.AcquisitionMode.SetIntValue(PySpin.AcquisitionMode_Continuous)

                info('setting frame rate enable to True')
                self.cam.AcquisitionFrameRateEnable.SetValue(True)
                info('setting frame rate to 1')
                self.cam.AcquisitionFrameRate.SetValue(20.0)
                info('setting TriggerMode Off')
                self.cam.TriggerMode.SetIntValue(PySpin.TriggerMode_Off)
                info('setting TriggerSource Line0')
                self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
                info('setting TriggerSource Software')
                self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)

    def convert_raw_to_image(self,rawdata):
        """
        """
        from lcp_video.analysis import mono12p_to_image, mono12packed_to_image
        from numpy import right_shift

        rawdata = rawdata[:self.img_len]
        print(f'rawdata = {rawdata.dtype}')
        if self.pixel_format == 'mono12p':
            image = mono12p_to_image(rawdata=rawdata,height=self.height,width=self.width)
        elif self.pixel_format == 'mono12packed':
            mask = self.conversion_mask
            mask8 = self.conversion_mask8
            image = mono12packe_to_image(rawdata=rawdata,height=self.height,width=self.width,mask=mask,mask8=mask8)
        elif self.pixel_format == 'mono16':
            image = right_shift(rawdata,4).reshape((self.height,self.width))
        elif self.pixel_format == 'mono12p_16':
            image = rawdata.reshape((self.height,self.width))
        return image


# Recording
#
    def record_dataset(self, root = '/mnt/data/', name = 'test', N_frames = 256, overwrite = False):
        import os
        from time import sleep, ctime, time
        self.recording_stop()
        sleep(0.5) #It is not clear to me what is the purpose of this sleep. It could to allow a last image to be collected. But that should take no more than "exposure time".
        self.queue.reset();
        if not os.path.exists(root):
            print('The root path did not exist and was created')
            os.mkdir(root)
        self.recording_root = root
        name = name.replace('_','-').replace('.','-')
        self.recording_init(N_frames,name,overwrite)
        self.recording_start()
        print(f'started at {ctime(time())} and awaiting trigger pulses. The name of the first chunk is {self.recording_filename}')

    def recording_init(self, N_frames = 1200, name = '',overwrite = False):
        """
        Initializes recording
        """
        self.recording_basefilename = self.recording_root+f'{self.name}_{name}'
        self.recording_chunk_pointer = 0
        filename = self.recording_basefilename + '_' + str(self.recording_chunk_pointer) + '.tmpraw.hdf5'
        self.recording_create_file(filename = filename, N_frames = N_frames, overwrite = overwrite)
        self.recording_Nframes = N_frames
        self.recording_pointer = 0

    def recording_create_file(self, filename, N_frames, overwrite = False):
        """
        creates hdf5 file on a drive prior to recording
        """
        from os.path import exists
        from h5py import File
        from time import time, sleep, ctime
        from numpy import zeros
        file_action = 'a'
        if exists(filename):
            if overwrite:
                info(f'The HDF5 file exists but will be overwritten. The HDF5 was created. The file name is {filename}')
                file_action = 'w'
            else:
                info(f' ----WARNING----')
                info(f' The HDF5 file exists. Please delete it first! or use parameter overwrite = True to force deletion and creating of new file.')
        else:
            info(f': The HDF5 was created. The file name is {filename}')
            with File(filename,file_action) as f:
                f.create_dataset('pixel format', data = self.pixel_format)
                f.create_dataset('exposure time', data = self.exposure_time)
                f.create_dataset('black level all', data = self.black_level['all'])
                f.create_dataset('black level analog', data = self.black_level['analog'])
                f.create_dataset('black level digital', data = self.black_level['digital'])
                f.create_dataset('image width', data = self.width)
                f.create_dataset('image height', data = self.height)
                f.create_dataset('image offset x', data = self.ROI_offset_x)
                f.create_dataset('image offset y', data = self.ROI_offset_y)
                f.create_dataset('gain', data = self.gain)
                f.create_dataset('time', data = ctime(time()))
                f.create_dataset('temperature', data = self.temperature)
                f.create_dataset('images', (N_frames,self.img_len), dtype = self.images_dtype, chunks = (1,self.img_len))
                f.create_dataset('timestamps_lab', (N_frames,) , dtype = 'float64')
                f.create_dataset('timestamps_camera', (N_frames,) , dtype = 'float64')
                f.create_dataset('frameIDs', (N_frames,) , dtype = 'int64')
        self.recording_filename = filename

    def record_once(self,filename,N):
        """
        records a sequence of N frames and save them in filename hdf5 file.
        """
        from h5py import File
        from time import time
        N = 1 #temporary work around. extract only 1 frame and save it. Got rid of for loop and now reading -1 index instead of 0.
        raws = self.queue.dequeue(N)

        if raws.shape[0] > 0:
            with File(filename,'a') as f:
                #for i in range(N):
                pointer = self.recording_pointer
                raw = raws[-1]#raw = raws[i]
                tlab,tcam,frameID = self.extract_timestamp_image(raw)
                f['images'][pointer] = raw[:self.img_len]
                f['timestamps_lab'][pointer] = tlab
                f['timestamps_camera'][pointer] = tcam
                f['frameIDs'][pointer] = frameID
                self.recording_pointer += 1
        else:
            warn(f'{self.name}: got empty array from deque with rear value in the queue of {self.queue.rear}')

    def recording_run(self):
        """
        records iamges to file in a loop.
        """
        from time import time,ctime
        from os import utime, rename
        from h5py import File
        while (self.recording):
            if self.recording_pointer == self.recording_Nframes:
                #create new file
                self.recording_chunk_pointer += 1
                if self.recording_chunk_pointer >= self.recording_chunk_maxpointer:
                    self.recording = False
                    break
                self.recording_pointer = 0
                rename(self.recording_basefilename + '_' + str(self.recording_chunk_pointer-1) + '.tmpraw.hdf5',self.recording_basefilename + '_' + str(self.recording_chunk_pointer-1) + '.raw.hdf5')
                f = File(self.recording_basefilename + '_' + str(self.recording_chunk_pointer-1) + '.raw.hdf5','r')
                timestamp = f['timestamps_lab'][0]
                utime(self.recording_basefilename + '_' + str(self.recording_chunk_pointer-1) + '.raw.hdf5',(timestamp,timestamp))

                filename = self.recording_basefilename + '_' + str(self.recording_chunk_pointer) + '.tmpraw.hdf5'
                Nframes = self.recording_Nframes
                self.recording_create_file(filename,Nframes)

            filename = self.recording_filename
            N = self.recording_N
            if (self.queue.length > N):
                self.record_once(filename = filename, N = N)
            else:
                sleep((N+3)*0.051)
        self.recording = False
        info(f'Recording Loop is finished')

    def recording_start(self):
        """
        a simple wrapper to start a recording_run loop in a separate thread.

        stops recording threads if such are active.
        resets queue to insure we save only new data.
        """
        from ubcs_auxiliary.multithreading import new_thread
        from time import time, sleep
        if self.recording:
            self.recording_stop()
            sleep(1)
        self.queue.reset()
        self.recording = True
        self.threads['recording'] = new_thread(self.recording_run)

    def recording_resume(self):
        """
        resumes images recording.

        Useful if the data collection was terminated due to
        """
        if self.recording:
            self.recording_stop()
            sleep(1)
        self.recording = True
        self.threads['recording'] = new_thread(self.recording_run)

    def recording_stop(self):
        """
        a wrapper that stops the recording loop
        """
        self.recording = False
        self.threads['recording'] = None


    def add_comment(self, comment):
        with open(self.recording_basefilename+'.txt','a') as f:
            f.write(f'time: {ctime(time())}' + '\n')
            f.write(f'chunk:  {self.recording_chunk_pointer}'+ '\n')
            f.write(f'chunk:  {comment}' + '\n')

    def extract_timestamp_image(self,raw):
        """
        Extracts header information from the

        returns timestamp_lab,timestamp_cam,frameid
        """
        pointer = self.img_len
        header_img = raw[pointer:pointer+64]
        timestamp_lab =  self.binarr_to_number(header_img)/1000000
        header_img = raw[pointer+64:pointer+128]
        timestamp_cam =  self.binarr_to_number(header_img)
        header_img = raw[pointer+128:pointer+192]
        frameid =  self.binarr_to_number(header_img)
        return timestamp_lab,timestamp_cam,frameid

    def binarr_to_number(self,vector):
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

    def bin_array(self,num, m):
        """
        Converts a positive integer num into an m-bit bit vector
        """
        from numpy import uint8, binary_repr,array
        return array(list(binary_repr(num).zfill(m))).astype(uint8)

def read_config_file(filename):
    import yaml
    import os
    flag =  os.path.isfile(filename)
    info(f'the file {filename} exists {flag}. Reading config file...')
    if flag:
        with open(filename,'r') as handle:
            config = yaml.safe_load(handle.read())  # (2)
    else:
        config = {}
    return config, flag

def get_PySpin_system():
    import PySpin
    system = PySpin.System.GetInstance()
    return system

if __name__ == '__main__':
    from tempfile import gettempdir
    print('sys.argv',sys.argv)
    if len(sys.argv)>1:
        camera = FlirCamera('temp')
        config_filename = sys.argv[1]
        config, flag = read_config_file(sys.argv[1])
        print('input parameters:',config,flag)
        logging.basicConfig(
            filename=gettempdir()+f"/{config['name']}_flir_camera_DL.log",
            level=logging.DEBUG,
            format="%(asctime)-15s|PID:%(process)-6s|%(levelname)-8s|%(name)s| module:%(module)s-%(funcName)s|message:%(message)s"
            )
        camera.init(config_filename)

    print('--- Example of usage ---')
    print("camera.recording_init(N_frames = 600, name = 'dataset-name', overwrite = False)")
    print('To read current settings on the camera')
    print("camera.read_current_setting()")
