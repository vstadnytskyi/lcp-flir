"""
DATAQ USB Bulk device level code
author: Valentyn Stadnytskyi
June 2018 - June 2018

1.0.0 - designed for usb Bulk protocol.
1.0.1 - dec is added to the communication

"""

from numpy import nan, mean, std, asarray, array, concatenate, delete, round, vstack, hstack, zeros, transpose, split
from time import time, sleep
import sys
import os.path
import struct
from pdb import pm
from time import gmtime, strftime

from struct import pack, unpack
import traceback

import logging
from logging import debug, info, warning, error
logging.getLogger(__name__).addHandler(logging.NullHandler())

from caproto.server import pvproperty, PVGroup, ioc_arg_parser, run
import caproto
from textwrap import dedent
import numpy as np
raw_image = np.zeros((int(3000*4096*1.5),1), dtype = 'uint8')

class Server(PVGroup):
    """
    An IOC with three uncoupled read/writable PVs

    Scalar PVs
    ----------

    Vectors PVs
    -----------
    AIO
    DIO

    """

    image = pvproperty(value=raw_image.flatten(), dtype=caproto.ChannelType.INT, read_only=True)

    timestamp_lab = pvproperty(value=0.0, units = 'time', dtype=float)
    timestamp_camera = pvproperty(value=0.0, units = 'time', dtype=float)
    frameID = pvproperty(value=0.0, units = 'count', dtype=int)


    DIO = pvproperty(value=0, units = 'counts')

    CH0 = pvproperty(value=0, units = 'counts')
    CH1 = pvproperty(value=0, units = 'counts')
    CH2 = pvproperty(value=0, units = 'counts')
    CH3 = pvproperty(value=0, units = 'counts')
    CH4 = pvproperty(value=0, units = 'counts')
    CH5 = pvproperty(value=0, units = 'counts')
    CH6 = pvproperty(value=0, units = 'counts')
    CH7 = pvproperty(value=0, units = 'counts')


    @image.startup
    async def image(self, instance, async_lib):
        # This method will be called when the server starts up.
        self.io_pull_queue = async_lib.ThreadsafeQueue()
        self.io_push_queue = async_lib.ThreadsafeQueue()
        camera.io_push_queue = self.io_push_queue

        # Loop and grab items from the response queue one at a time
        while True:
            io_dict = await self.io_push_queue.async_get()
            # Propagate the keypress to the EPICS PV, triggering any monitors
            # along the way
            for key in list(io_dict.keys()):
                if key == 'image':
                    await self.image.write(io_dict[key])
                elif key == 'timestamp_lab':
                    await self.timestamp_lab.write(io_dict[key])
                elif key == 'timestamp_camera':
                    await self.timestamp_camera.write(io_dict[key])
                elif key == 'frameID':
                    await self.frameID.write(io_dict[key])
                elif key == 'DIO':
                    await self.DIO.write(io_dict[key])
                elif key == 'CH0':
                    await self.CH0.write(io_dict[key])
                elif key == 'CH1':
                    await self.CH1.write(io_dict[key])
                elif key == 'CH2':
                    await self.CH2.write(io_dict[key])
                elif key == 'CH3':
                    await self.CH3.write(io_dict[key])
                elif key == 'CH4':
                    await self.CH4.write(io_dict[key])
                elif key == 'CH5':
                    await self.CH5.write(io_dict[key])
                elif key == 'CH6':
                    await self.CH6.write(io_dict[key])
                elif key == 'CH7':
                    await self.CH7.write(io_dict[key])

def start(config_filename):
    from lcp_flir import flir_camera_DL
    camera = flir_camera_DL.FlirCamera()
    camera.init(config_filename)
    return camera

if __name__ == "__main__": #for testing
    from tempfile import gettempdir
    import sys

    config_filename = '/net/femto/C/All Projects/LaserLab/Software/instruments/Aerosols/config_bottom_12bit_server.conf'
    camera = start(config_filename = config_filename)
    camera.broadcast_frames = True

    from caproto import config_caproto_logging
    config_caproto_logging(file=gettempdir()+'/camera_DLS.log', level='DEBUG')


    ioc_options, run_options = ioc_arg_parser(
        default_prefix=f"{camera.config['network_name']}:",
        desc=dedent(Server.__doc__))
    ioc = Server(**ioc_options)
    run(ioc.pvdb, **run_options)
