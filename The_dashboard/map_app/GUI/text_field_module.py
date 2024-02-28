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
from .gridstack_handler import grid_stack
from .analysis_page_abstract import analysis_abstract
pd.options.mode.chained_assignment = None 
logging.basicConfig( level=logging.ERROR,force=True)

pn.extension('floatpanel')
pn.extension('gridstack')
pn.extension('tabulator')
pn.extension('plotly')
pn.extension('texteditor')
pn.extension(notifications=True)
pn.extension(design='bootstrap', template='material' )


class text_field_manger: 
    def __init__(self,grid_area_obj=None) -> None:
        self.grid_area_obj = grid_area_obj
        self.text_fields = []
        self.buttons_column = pn.Column()
        self.cur_text_fields = 0


    def set_grid_area_obj(self,obj):
        self.grid_area_obj = obj

    def check_button(self,event):
        txt_name = event.obj.name
        #show_value = event.obj.value
        if event.obj.clicks%2==0:
            event.obj.button_style='outline'
            show_value= False
        else:
            show_value = True
            event.obj.button_style='solid'
        try:
            if show_value:
                txt_field_index = int(txt_name[1:]) -1
                self.add_text_field(self.text_fields[txt_field_index])
                
            else:
                self.grid_area_obj.remove_chart(txt_name)
        except:
            pass
        
    def add_text_field(self,txt_field):
        self.grid_area_obj.add_chart(txt_field)

    def bend_text_fields_buttons(self,new_button):
        new_button.param.watch(self.check_button,'value') 
    
    def get_buttons_column(self):
        return self.buttons_column
    

    def get_fields_colum(self):
        return self.text_fields
    
    def create_new_text_field(self,place_holder_text ='Write your description'):
        format_toolbar = [
                                [{ 'header': 1 }, { 'header': 2 }],
                                [{ 'list': 'ordered'}, { 'list': 'bullet' }],
                                ['link', 'image']
                                ]
        self.cur_text_fields+=1
        new_text_field = pn.widgets.TextEditor(mode='bubble', value=place_holder_text, margin=(40, 0, 0, 0), height=200, width=400,name=f'T{self.cur_text_fields}',toolbar = format_toolbar )
        new_button = pn.widgets.Button(button_type='primary', button_style='solid', align='center',name=f'T{self.cur_text_fields}',value=True,description=f'Text Field {self.cur_text_fields}')
        self.text_fields.append(new_text_field)
        self.buttons_column.append(new_button)
        self.bend_text_fields_buttons(new_button)
        self.add_text_field(new_text_field)

    
