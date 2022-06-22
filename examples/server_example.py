from logging import debug, info, warning, error
import logging
import os
logging.getLogger(__name__).addHandler(logging.NullHandler())

import lcp_flir
print('import lcp_flir')
from lcp_flir.flir_camera_DL import read_config_file, FlirCamera
from tempfile import gettempdir
#
config_filename =  'C:/Users/AR-VR lab W1/Documents/Valentyn/custom_python_libraries/instrumentation/flir-camera/config_SN14120164_12bit.conf'
print(f'Does config file exist? {os.path.isfile(config_filename)}')
config, flag = lcp_flir.flir_camera_DL.read_config_file(config_filename)
print('lcp_flir.flir_camera_DL.read_config_file')
name2 = config['name']
logging.basicConfig(
     filename=gettempdir()+f"/test_flir_camera_DL.log",
     level=logging.INFO,
     format="%(asctime)-15s|PID:%(process)-6s|%(levelname)-8s|%(name)s| module:%(module)s-%(funcName)s|message:%(message)s"
     )
info('test run')

camera = FlirCamera()
camera.init(config_filename = config_filename)
camera.broadcast_frames = True

from lcp_flir.device_level_server import Server

from caproto.server import pvproperty, PVGroup, ioc_arg_parser, run
import caproto
from caproto import config_caproto_logging
from textwrap import dedent
config_caproto_logging(file=gettempdir()+'/camera_DLS.log', level='INFO')


ioc_options, run_options = ioc_arg_parser(
    default_prefix=f"{camera.config['network_name']}:",
    desc=dedent(Server.__doc__))
ioc = Server(**ioc_options)
ioc.device = camera
run(ioc.pvdb, **run_options)
