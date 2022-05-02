from logging import debug, info, warning, error
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

import lcp_flir
print('import lcp_flir')
from lcp_flir.flir_camera_DL import read_config_file, FlirCamera
from tempfile import gettempdir
#
config_filename =  'config_mono_12bit.conf'
config, flag = lcp_flir.flir_camera_DL.read_config_file(config_filename)
print('lcp_flir.flir_camera_DL.read_config_file')
name2 = config['name']
logging.basicConfig(
     filename=gettempdir()+f"/test_flir_camera_DL.log",
     level=logging.DEBUG,
     format="%(asctime)-15s|PID:%(process)-6s|%(levelname)-8s|%(name)s| module:%(module)s-%(funcName)s|message:%(message)s"
     )
info('test run')
#     )
#
camera = FlirCamera()
camera.init(config_filename = config_filename)