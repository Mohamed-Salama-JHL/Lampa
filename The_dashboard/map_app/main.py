import panel as pn
from bokeh.embed import server_document
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from GUI.map_page import createApp
from bokeh.settings import settings


settings.resources = 'cdn'

pn.serve(createApp,
        port=5000, allow_websocket_origin=["*"],
        address="0.0.0.0", show=False,
        websocket_max_message_size=150*1024*1014,
        http_server_kwargs={'max_buffer_size': 150*1024*1014})