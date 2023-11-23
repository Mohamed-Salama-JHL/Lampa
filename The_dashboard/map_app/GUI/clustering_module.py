import numpy as np
import param
import panel as pn
import numpy as np
import hvplot.dask
import pandas as pd
import folium
import panel as pn
import hvplot.pandas
import pandas as pd
import numpy as np
from io import StringIO
from urllib.request import urlopen
import json
import plotly.express as px
import plotly.graph_objs as go
import threading
from panel.theme import Native
from .data_handler import *
from .geo_data_handler import * 
from .styles import *
import copy
from panel.layout.gridstack import GridStack
import logging
from sklearn.preprocessing import MinMaxScaler

from .gridstack_handler import grid_stack
from .analysis_page_abstract import analysis_abstract

logging.basicConfig( level=logging.ERROR,force=True)

pn.extension('floatpanel')
pn.extension('gridstack')

pn.extension('tabulator')
pn.extension('plotly')
#pn.extension(loading_spinner='dots', loading_color='#00aa41')
pn.extension(notifications=True)


class clustering_module(analysis_abstract):
    
    def __init__(self,dataset,dataset_controls) -> None:
        self.dataset = dataset
        self.dataset_controls = dataset_controls
        self.algorithms_list = ['k-means','DBscan']
        self.feature_reduction_algorithms_list = ['PCA']


    def get_page(self):
        pass

    def get_dataset_new(self):
        pass

    def get_end__button(self):
        pass

    def create_sidebar(self):
        self.create_sidebar()
        self.controls = pn.WidgetBox(self.algo_settings_card,self.data_settings_card,height=1100, width=330,styles={ "background":"#FAFAFA"},visible=False)
    
    def create_main_area(self):
        pass

    def create_algorithm_settings_component(self):
        self.select_algorithm = pn.widgets.Select(name='Algorithm select', options=self.algorithms_list)
        self.select_num_clusters = pn.widgets.EditableIntSlider(name='Number of clusters', start=0, end=15, step=1, value=3)
        self.select_feature_reduction_algorithm_selection = pn.widgets.Select(name='Algorithm select', options=self.feature_reduction_algorithms_list)
        self.algo_settings_card = pn.Card(self.select_algorithm, self.select_feature_reduction_algorithm_selection,self.select_num_clusters, title='Card', styles={'background': 'WhiteSmoke'})

    def create_dataset_setting_component(self):
        self.select_columns = pn.widgets.MultiChoice(name= 'Columns needed' ,options=self.dataset.columns,  visible=False)
        self.check_normalization =  pn.widgets.Switch(name='Normalization')
        self.data_settings_card = pn.Card(self.select_columns,self.check_normalization, title='Card', styles={'background': 'WhiteSmoke'})


    def get_updated_dataset_button(self):
        pass

    def updated_dataset(self,new_dataset):
        pass

    def run_clustering(self):
        self.dataset_preprocessing()
        if self.check_normalization.value:
            self.normalize_dataset()
        if self.select_algorithm.value=='k-means':
            pass
        elif self.select_algorithm.value=='DBscan':
            pass

    
    def dataset_preprocessing(self):
        dataset = self.dataset[self.select_columns.value]
        non_numeric_columns = dataset.select_dtypes(exclude=['number']).columns

        if not non_numeric_columns.empty:
            self.working_dataset = pd.get_dummies(dataset, columns=non_numeric_columns)

        else:
            self.working_dataset = self.dataset

        if self.check_normalization.value:
            self.normalize_dataset()


    def normalize_dataset(self):
        scaler = MinMaxScaler()
        self.working_dataset = scaler.fit_transform(self.working_dataset)



    def update_charts(self):
        pass
    

