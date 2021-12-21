import logging
from logging import debug, info, warning, error
logging.getLogger(__name__).addHandler(logging.NullHandler())

from caproto.threading.client import Context
ctx = Context()
prefix = 'camera_top'
image, frameID = ctx.get_pvs(f'{prefix}:image',f'{prefix}:frameID')

if __name__ == '__main__':
    import logging
    logging.basicConfig(filename=gettempdir()+'/camera_top_DLC.log',
                    level=logging.DEBUG,
                    format="%(asctime)-15s|PID:%(process)-6s|%(levelname)-8s|%(name)s|module:%(module)s-%(funcName)s|message:%(message)s")
