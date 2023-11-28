import panel as pn
from bokeh.embed import server_document
from GUI.map_page import createApp
from bokeh.settings import settings
import logging
import sys

logging.basicConfig( level=logging.ERROR,force=True)
'''
# Create a custom logging handler that redirects stdout to the logger
class StreamToLogger:
    def __init__(self, logger, log_level=logging.DEBUG):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

# Redirect stdout and stderr to the custom logging handler
sys.stdout = StreamToLogger(logging.getLogger('stdout'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('stderr'), logging.ERROR)
'''
settings.resources = 'cdn'

pn.serve(createApp,
        port=5000, allow_websocket_origin=["*"],log_level = 'ERROR',
        address="0.0.0.0", show=False,
        websocket_max_message_size=150*1024*1014,
        http_server_kwargs={'max_buffer_size': 150*1024*1014})