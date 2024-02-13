import numpy as np
import param
import panel as pn
import numpy as np
import hvplot.dask
import pandas as pd
import panel as pn
import hvplot.pandas
import pandas as pd
import numpy as np
from io import StringIO
from panel.theme import Native
import plotly.express as px
import plotly.graph_objs as go
import logging
from .num_input import *
from .gridstack_handler import grid_stack
from .analysis_page_abstract import analysis_abstract
pd.options.mode.chained_assignment = None 
logging.basicConfig( level=logging.ERROR,force=True)

pn.extension('floatpanel')
pn.extension('gridstack')
pn.extension('tabulator')
pn.extension('plotly')
pn.extension(notifications=True)
pn.extension(design='bootstrap', template='material' )


class text_field_manger:
    def __init__(self,grid_area_obj=None) -> None:
        self.grid_area_obj = grid_area_obj
        self.text_fields = []
        self.buttons_column = pn.Column()
        self.cur_text_fields = 0

    def bend_text_fields_buttons(self):
        pass
    
    def get_buttons_column(self):
        return self.buttons_column
    

    def get_fields_colum(self):
        return self.text_fields
    
    def get_text_field(self,place_holder_text ='Write your description'):
        self.cur_text_fields+=1
        new_text_field = pn.widgets.TextEditor(mode='bubble', value=place_holder_text, margin=(40, 0, 0, 0), height=200, width=400,name=f'T{self.cur_text_fields}')
        new_button = pn.widgets.Toggle(button_type='light', button_style='solid', align='center',name=f'T{self.cur_text_fields}')

        return new_text_field
    
