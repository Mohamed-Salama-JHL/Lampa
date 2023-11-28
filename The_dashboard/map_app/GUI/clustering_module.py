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
from sklearn.cluster import KMeans,DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objs as go
import threading
from panel.theme import Native

import copy
from panel.layout.gridstack import GridStack
import logging
from sklearn.preprocessing import MinMaxScaler

from gridstack_handler import grid_stack
from analysis_page_abstract import analysis_abstract

pd.options.mode.chained_assignment = None 
logging.basicConfig( level=logging.ERROR,force=True)

pn.extension('floatpanel')
pn.extension('gridstack')
pn.extension('tabulator')
pn.extension('plotly')
pn.extension(notifications=True)

class clustering_module:
    
    def __init__(self,dataset,dataset_controls) -> None:
        self.dataset = dataset
        self.working_dataset = None
        self.output_dataset = None
        self.reduction_dataset = None
        self.dataset_controls = dataset_controls
        self.algorithms_list = ['k-means','DBscan']
        self.feature_reduction_algorithms_list = ['PCA','t-SNE']
        self.clustering_page = None
    
    def get_updated_dataset_button(self):
        return self.update_results_button

    def set_dataset(self,new_dataset):
        self.dataset = new_dataset
    
    def get_page(self):
        if self.clustering_page == None:
            self.create_page()
        return self.clustering_page
    
    def create_page(self):
        self.create_sidebar()
        self.create_main_area()
        self.bend_components_actions()
        self.clustering_page = pn.Column(
            pn.Row(
                self.clustering_controls,
                self.clustering_main_area,
                sizing_mode='stretch_width'
            ),
            sizing_mode='stretch_both')
    
    def create_sidebar (self):
        self.create_algorithm_settings_component()
        self.create_dataset_setting_component()
        self.create_control_buttons()
        self.create_general_widgets()
        self.clustering_controls = pn.Card(self.algo_settings_card,self.data_settings_card,self.controls_buttons_row,self.loading,height=1100, width=330,styles={ "background":"#FAFAFA"})
    
    def get_grid_stack(self,charts_dict):
        self.grid_stack_handler = grid_stack(charts_dict)
        return self.grid_stack_handler.get_gridstack()
    
    def create_main_area(self):
        self.elbow_chart = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='Elbow chart', margin=2),scroll=False)
        self.data_points_2d = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='datapoint chart', margin=2),scroll=False)
        self.clustering_main_area = self.get_grid_stack([self.elbow_chart,self.data_points_2d])
        self.clustering_main_area.visible = True

    def create_algorithm_settings_component(self):
        self.select_algorithm = pn.widgets.Select(name='Clustering algorithms', options=self.algorithms_list, value ='k-means' )
        self.select_num_clusters = pn.widgets.EditableIntSlider(name='Number of clusters', start=1, end=15, step=1, value=3)
        self.select_feature_reduction_algorithm_selection = pn.widgets.Select(name='Feature reduction algorithms', options=self.feature_reduction_algorithms_list)
        self.dbscan_eps = pn.widgets.FloatInput(name='eps value', value=0.5, step=1e-1, start=0, end=1000,visible = False)
        self.dbscan_eps_min_samples = pn.widgets.IntInput(name='Min number of samples', start=1, end=1000, step=1, value=5,visible = False)
        self.algo_settings_card = pn.Card(self.select_algorithm, self.select_feature_reduction_algorithm_selection,self.select_num_clusters,
                                          self.dbscan_eps,self.dbscan_eps_min_samples, title='Card', styles={'background': 'WhiteSmoke'})

    def create_dataset_setting_component(self):
        self.select_columns = pn.widgets.MultiChoice(name= 'Columns needed' ,options=list(self.dataset.columns))
        self.check_normalization = pn.widgets.Switch(name='Normalization',margin = 0)
        norm_name = pn.widgets.StaticText(value='Normalization: ',margin = 0)
        self.data_settings_card = pn.Card(self.select_columns,pn.Column('Normalization: ',pn.widgets.Switch(name='Normalization')), title='Card', styles={'background': 'WhiteSmoke'})

    def create_control_buttons(self):
        self.run_clustering_button = pn.widgets.Button(name='Run Clustering', button_type='primary')
        self.update_results_button =  pn.widgets.Button(name='Update Results', button_type='primary')
        self.controls_buttons_row = pn.Row(self.run_clustering_button,self.update_results_button)
    
    def create_general_widgets(self):
        self.loading = pn.indicators.LoadingSpinner(value=True, size=20, name='Loading...', visible=False)
    
    def run_clustering(self,event=None):
        self.start_clustering()    
        self.dataset_preprocessing()
        distortions = None
        if self.select_algorithm.value=='k-means':
            number_clusters = int(self.select_num_clusters.value)
            distortions = self.run_kmeans(number_clusters)
        elif self.select_algorithm.value=='DBscan':
            self.run_dbscan()
        self.feature_reduction()
        self.create_charts_objects(distortions)
        self.end_clustering()
    
    def start_clustering(self):
        self.loading.visible= True
        self.run_clustering_button.disabled = True
    def end_clustering(self):
        self.loading.visible= False
        self.run_clustering_button.disabled = False
    
    def dataset_preprocessing(self):
        self.output_dataset = self.dataset[self.select_columns.value]
        non_numeric_columns = self.output_dataset.select_dtypes(exclude=['number']).columns

        if not non_numeric_columns.empty:
            self.working_dataset = pd.get_dummies(self.output_dataset, columns=non_numeric_columns)

        else:
            self.working_dataset = self.output_dataset

        if self.check_normalization.value:
            self.normalize_dataset()
    
    def normalize_dataset(self):
        scaler = MinMaxScaler()
        self.working_dataset = scaler.fit_transform(self.working_dataset)

    def run_kmeans(self,clusters_num=None):
        distortions = []
        for k in range(1, 16):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.working_dataset)
            distortions.append(kmeans.inertia_)
            if k==clusters_num:
                self.output_dataset.loc[:, 'Cluster'] = kmeans.fit_predict(self.working_dataset)
        return distortions 
        
    def run_dbscan(self):
        eps = self.dbscan_eps.value 
        min_samples = self.dbscan_eps_min_samples.value 

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.output_dataset.loc[:, 'Cluster'] = dbscan.fit_predict(self.working_dataset)

    def feature_reduction(self):
        if self.select_feature_reduction_algorithm_selection.value =='PCA':
            self.run_pca_reduction()
        elif  self.select_feature_reduction_algorithm_selection.value == 't-SNE':
            self.run_tsne_reduction()
    
    def run_pca_reduction(self,number_components = 2):
        self.reduction_dataset = self.output_dataset[self.select_columns.value]
        pca = PCA(n_components=number_components)
        pca_result = pca.fit_transform(self.reduction_dataset)
        self.reduction_dataset = pd.DataFrame(data=pca_result, columns=['C1', 'C2'])
        self.reduction_dataset['Cluster'] = self.output_dataset['Cluster']
        self.reduction_dataset= pd.concat([self.dataset, self.reduction_dataset], axis=1)

    def run_tsne_reduction(self,number_components = 2):
        self.reduction_dataset = self.output_dataset[self.select_columns.value]
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(self.reduction_dataset)
        self.reduction_dataset = pd.DataFrame(data=tsne_results, columns=['C1', 'C2'])
        self.reduction_dataset['Cluster'] = self.output_dataset['Cluster']
        self.reduction_dataset= pd.concat([self.dataset, self.reduction_dataset], axis=1)

    def create_charts_objects(self,distortions=None):
        elbow_fig = None
        self.reduction_dataset["Cluster"] = self.reduction_dataset["Cluster"].astype(str)
        output_fig = px.scatter(self.reduction_dataset, x='C1', y='C2', color='Cluster',hover_data=self.reduction_dataset.columns,
                 title=f'{self.select_feature_reduction_algorithm_selection.value} Scatter Plot with Cluster Colors', labels={'color': 'Cluster'},
                 template="plotly_white").update_layout(margin=dict(l=20, r=20, t=50, b=5),)
        if distortions!=None:
            elbow_fig = px.line(x=range(1, 16), y=distortions, title='Elbow Method for Optimal k',
              labels={'x': 'Number of Clusters (k)', 'y': 'Distortion'},
              markers=True, line_shape="linear", template="plotly_white").update_layout(margin=dict(l=20, r=20, t=50, b=5),)
            elbow_fig.layout.autosize = True
        output_fig.layout.autosize = True
        self.update_charts(output_fig,elbow_fig)
            
    def update_charts(self,output_fig,elbow_fig=None):
        self.data_points_2d.clear()
        self.data_points_2d.append(pn.pane.Plotly(output_fig,name='datapoint chart',  margin=2)) 
        if elbow_fig!=None:
            self.elbow_chart.clear()
            self.elbow_chart.append(pn.pane.Plotly(elbow_fig,name='Elbow chart',  margin=2))

    def show_kmeans_settings(self):
        self.select_num_clusters.visible = True
        self.dbscan_eps_min_samples.visible = False 
        self.dbscan_eps.visible = False
        self.grid_stack_handler.add_chart(self.elbow_chart)
            
    def show_dbscan_settings(self):
        self.select_num_clusters.visible = False
        self.dbscan_eps_min_samples.visible = True 
        self.dbscan_eps.visible = True
        self.grid_stack_handler.remove_chart(self.elbow_chart.name)
    
    def algo_settings_show(self,event):
        if self.select_algorithm.value == 'k-means':
            self.show_kmeans_settings()
        elif self.select_algorithm.value == 'DBscan':
            self.show_dbscan_settings()
    
    def bend_components_actions(self):
        self.run_clustering_button.param.watch(self.run_clustering,'value')
        self.select_algorithm.param.watch(self.algo_settings_show,'value')
            