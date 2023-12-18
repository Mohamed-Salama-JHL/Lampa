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
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objs as go
import threading
from panel.theme import Native
from prince import MCA
import copy
from panel.layout.gridstack import GridStack
import logging
from sklearn.preprocessing import MinMaxScaler
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

class clustering_module:
    
    def __init__(self,dataset) -> None:
        self.dataset = dataset#.reset_index()
        self.working_dataset = None
        self.output_dataset = None
        self.reduction_dataset = None
        self.algorithms_list = ['k-means','DBscan','Hierarchical Clustering']
        self.feature_reduction_algorithms_list = ['PCA','t-SNE']
        self.clustering_page = None
        self.actual_fig_flag = False
        self.main_page_created = False
    
    def get_updated_dataset_button(self):
        return self.update_results_button

    def set_dataset(self,new_dataset):
        #new_dataset=new_dataset.reset_index()
        self.dataset = new_dataset
    
    def get_page(self):
        if self.clustering_page == None:
            self.create_page()
        return self.clustering_page
    
    def get_controls(self):
        if self.clustering_page == None:
            self.create_page()
        return self.clustering_controls
    
    def get_main_area(self):
        if self.clustering_page == None:
            self.create_page()
        
        return self.clustering_main_area
    
    def create_page(self):
        self.create_sidebar()
        self.create_main_area()
        self.bend_components_actions()
        self.clustering_page = pn.Column(
            pn.Row(
                self.clustering_main_area,
                self.clustering_controls,
                #sizing_mode='stretch_width'
            ),
            sizing_mode='stretch_both',name='Clustering')
    
    def create_sidebar (self):
        self.create_algorithm_settings_component()
        self.create_dataset_setting_component()
        self.create_control_buttons()
        self.create_general_widgets()
        self.clustering_controls = pn.Card(pn.Column(self.algo_settings_card,self.data_settings_card,self.controls_buttons_row,self.loading), title="<h1 style='font-size: 15px;'>Clustering</h1>", styles={"border": "none", "box-shadow": "none"})
    
    def get_grid_stack(self,chart_list):
        self.grid_stack_handler = grid_stack(chart_list,min_width=950,min_height=2000,ncols=6,nrows = 15, name='Clustering')
        return self.grid_stack_handler.get_gridstack()
    
    def create_main_area(self):
        
        self.elbow_chart = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='Elbow chart', margin=2,width=600),scroll=False,visible = False)
        self.data_points_2d = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='datapoint chart', margin=2,width=600),scroll=False,visible = False)
        self.data_points_2d_actual = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='datapoint chart', margin=2,width=600),scroll=False)
        chart_list= [self.data_points_2d,self.elbow_chart] 
        self.clustering_main_area = self.get_grid_stack(chart_list)
        self.clustering_main_area.visible = True

        #self.test_column = pn.Column(self.elbow_chart,name='Clustering')


    def create_main_area_alter(self):
        if not self.main_page_created:
            self.main_page_created = True
            self.elbow_chart = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='Elbow chart', margin=2),scroll=False,visible = False)
            self.data_points_2d = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='datapoint chart', margin=2),scroll=False,visible = False)
            self.data_points_2d_actual = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='datapoint chart', margin=2),scroll=False)
            self.clustering_main_area = pn.Column(self.data_points_2d,self.elbow_chart,self.data_points_2d_actual,name='Clustering',visible = True)

    def create_algorithm_settings_component(self):
        self.select_algorithm = pn.widgets.Select(name='Clustering algorithms', options=self.algorithms_list, value ='k-means' )
        self.select_num_clusters = number_input(type='int',title='Number of clusters: ',tooltip_str='Number of clusters which <br>will be used in the charts', start=1, end=15, step=1, value=3)
        self.select_feature_reduction_algorithm_selection = pn.widgets.Select(name='Feature reduction algorithms', options=self.feature_reduction_algorithms_list)
        self.dbscan_eps = number_input(type='float',title='eps value: ',tooltip_str='The maximum distance <br>between two samples for<br> one to be considered <br> as in the neighborhood <br> of the other.', value=0.5, step=1e-1, start=0, end=100000000,visible=False)
        self.dbscan_eps_min_samples = number_input(type='int',title='Min number of Samples',tooltip_str='The number of samples<br> (or total weight) in<br> a neighborhood for <br>a point to be considered<br> as a core point.<br> This includes the<br> point itself. ', start=1, end=1000000, step=1, value=5,visible=False)
        self.algo_settings_card = pn.Column(self.select_algorithm, self.select_feature_reduction_algorithm_selection,
                                            self.select_num_clusters.get_item(),
                                          self.dbscan_eps.get_item(),self.dbscan_eps_min_samples.get_item())#, title="<h1 style='font-size: 15px;'>Algorithm settings</h1>", styles={"border": "none", "box-shadow": "none"}

    def create_dataset_setting_component(self):
        numric_columns = list(self.dataset.select_dtypes(include=['number']).columns)
        self.select_clustering_columns = pn.widgets.MultiChoice(name= 'Clustering Columns' ,options=numric_columns)
        self.select_datapoints_columns = pn.widgets.MultiChoice(name= 'Tooltip datapoints Columns' ,options=list(self.dataset.columns))
        self.check_normalization = pn.widgets.Switch(name='Normalization',margin = 0)
        norm_name = pn.widgets.StaticText(value='Normalization: ',margin = 0)
        self.data_settings_card = pn.Column(self.select_clustering_columns,self.select_datapoints_columns,pn.Column(norm_name,pn.widgets.Switch(name='Normalization')))#, title="<h1 style='font-size: 15px;'>Dataset settings</h1>", styles={"border": "none", "box-shadow": "none"})

    def create_control_buttons(self):
        self.run_clustering_button = pn.widgets.Button(name='Run Clustering', button_type='primary')
        self.update_results_button =  pn.widgets.Button(name='Update Results', button_type='primary')
        self.download_clustered_data_button = pn.widgets.FileDownload(callback=pn.bind(self.get_clustered_data_io), filename='Clustered_data.csv', label = 'Download Dataset',align = 'center',button_style='outline',button_type='primary',height=40 )

        self.controls_buttons_row = pn.Column(pn.Row(self.run_clustering_button,self.update_results_button,sizing_mode='stretch_width', margin=(0, 30, 0, 30)),self.download_clustered_data_button,sizing_mode='stretch_width')
    
    def create_general_widgets(self):
        self.loading = pn.indicators.LoadingSpinner(value=True, size=20, name='Loading...', visible=False)
    
    def run_clustering(self,event=None):
        self.start_clustering()    
        self.dataset_preprocessing()
        distortions = None
        if self.select_algorithm.value=='k-means':
            number_clusters = int(self.select_num_clusters.get_value())
            distortions = self.run_kmeans(number_clusters)
        elif self.select_algorithm.value=='DBscan':
            self.run_dbscan()
        elif self.select_algorithm.value == 'Hierarchical Clustering':
            number_clusters = int(self.select_num_clusters.get_value())
            self.run_agglomerativeClustering(number_clusters)
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
        self.clustering_features_list = self.select_clustering_columns.value if len(self.select_clustering_columns.value)>=1 else self.select_clustering_columns.options
        self.output_dataset = self.dataset[self.clustering_features_list]
        self.missing_values_handling()
        non_numeric_columns = self.output_dataset.select_dtypes(exclude=['number']).columns
        self.categorical_flag = False
        
        if not non_numeric_columns.empty:
            self.working_dataset = pd.get_dummies(self.output_dataset, columns=non_numeric_columns)
            self.categorical_flag = True

        else:
            self.working_dataset = self.output_dataset

        if self.check_normalization.value:
            self.normalize_dataset()
    
    def normalize_dataset(self):
        scaler = MinMaxScaler()
        self.working_dataset = scaler.fit_transform(self.working_dataset)


    def missing_values_handling(self):
        self.output_dataset.dropna(inplace=True)

    def run_kmeans(self,clusters_num=None):
        distortions = []
        for k in range(1, 16):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) 
            kmeans.fit(self.working_dataset)
            distortions.append(kmeans.inertia_)
            if k==clusters_num:
                self.output_dataset.loc[:, 'Cluster'] = kmeans.fit_predict(self.working_dataset)
        return distortions 
    
    def run_agglomerativeClustering(self,clusters_num=None):

        agg_clustering = AgglomerativeClustering(n_clusters=clusters_num) 
        self.output_dataset.loc[:, 'Cluster'] = agg_clustering.fit_predict(self.working_dataset)

        
    def run_dbscan(self):
        eps = self.dbscan_eps.get_value() 
        min_samples = self.dbscan_eps_min_samples.get_value() 

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.output_dataset.loc[:, 'Cluster'] = dbscan.fit_predict(self.working_dataset)


    def feature_reduction(self):
        if self.select_feature_reduction_algorithm_selection.value =='PCA':
            if not self.categorical_flag:
                self.run_pca_reduction()
            else:
                self.run_mca_reduction()
        
        elif  self.select_feature_reduction_algorithm_selection.value == 't-SNE':
            self.run_tsne_reduction()

    def run_pca_reduction(self,number_components = 2):
        self.reduction_dataset = self.output_dataset[self.clustering_features_list]
        pca = PCA(n_components=number_components)
        pca_result = pca.fit_transform(self.reduction_dataset)
        self.reduction_dataset = pd.DataFrame(data=pca_result, columns=['C1', 'C2'], index=self.output_dataset.index)
        self.reduction_dataset['Cluster'] = self.output_dataset['Cluster']
        if len(self.select_datapoints_columns.value)>0:
            self.reduction_dataset= self.reduction_dataset.merge(self.dataset[self.select_datapoints_columns.value], left_index=True, right_index=True,how='left')

        
    def run_mca_reduction(self,number_components = 2):
        self.reduction_dataset = self.output_dataset[self.clustering_features_list]
        mca = MCA(n_components=number_components)
        mca_result = mca.fit_transform(self.reduction_dataset)
        self.reduction_dataset = pd.DataFrame(data=mca_result, columns=['C1', 'C2'], index=self.output_dataset.index)
        self.reduction_dataset['Cluster'] = self.output_dataset['Cluster']
        if len(self.select_datapoints_columns.value)>0:
            self.reduction_dataset= pd.concat([self.dataset[self.select_datapoints_columns.value], self.reduction_dataset], axis=1)


    def run_tsne_reduction(self,number_components = 2):
        self.reduction_dataset = self.working_dataset
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(self.reduction_dataset)
        self.reduction_dataset = pd.DataFrame(data=tsne_results, columns=['C1', 'C2'], index=self.output_dataset.index)
        self.reduction_dataset['Cluster'] = self.output_dataset['Cluster']
        if len(self.select_datapoints_columns.value)>0:

            self.reduction_dataset= pd.concat([self.dataset[self.select_datapoints_columns.value], self.reduction_dataset], axis=1)

    def create_charts_objects(self,distortions=None):
        elbow_fig = None
        actual_fig = None
        self.elbow_chart.visible = True
        self.data_points_2d.visible = True
        self.reduction_dataset["Cluster"] = self.reduction_dataset["Cluster"].astype(str)

        output_fig = px.scatter(self.reduction_dataset, x='C1', y='C2', color='Cluster',hover_data=self.reduction_dataset.columns,
                 title=f'{self.select_feature_reduction_algorithm_selection.value} Scatter Plot with Cluster Colors', labels={'color': 'Cluster'},
                 template="plotly_white").update_layout(margin=dict(l=20, r=20, t=50, b=5),)
        if distortions!=None:
            distortions_scaled = (distortions - np.min(distortions)) / (np.max(distortions) - np.min(distortions))
            elbow_fig = px.line(x=[i+1 for i in range(15)], y=distortions_scaled, title='Elbow Method for Optimal k',
              labels={'x': 'Number of Clusters (k)', 'y': 'Distortion scaled'},
              markers=True, line_shape="linear", template="plotly_white").update_layout(margin=dict(l=20, r=20, t=50, b=5),)
            elbow_fig.layout.autosize = True

        if len(self.select_clustering_columns.value)==2:
            self.output_dataset["Cluster"] = self.output_dataset["Cluster"].astype(str)
            actual_fig = px.scatter(self.output_dataset, x=self.select_clustering_columns.value[0], y=self.select_clustering_columns.value[1], color='Cluster',
                 title=f'{self.select_feature_reduction_algorithm_selection.value} Scatter Plot with Cluster Colors', labels={'color': 'Cluster'},
                 template="plotly_white").update_layout(margin=dict(l=20, r=20, t=50, b=5),)
            actual_fig.layout.autosize= True
        output_fig.layout.autosize = True
        self.update_charts(output_fig,elbow_fig,actual_fig)
            
    def update_charts(self,output_fig,elbow_fig=None,actual_fig=None):
        self.data_points_2d.clear()
        self.data_points_2d.append(pn.pane.Plotly(output_fig,name='datapoint chart',  margin=2)) 
        if elbow_fig!=None:
            self.elbow_chart.clear()
            self.elbow_chart.append(pn.pane.Plotly(elbow_fig,name='Elbow chart',  margin=2))
        if actual_fig!=None:
            self.data_points_2d_actual.clear()
            self.data_points_2d_actual.append(pn.pane.Plotly(actual_fig,name='datapoint chart',  margin=2)) 
            if not self.actual_fig_flag:
                self.grid_stack_handler.add_chart(self.data_points_2d_actual)
                self.actual_fig_flag = True
            
        else:
            self.grid_stack_handler.remove_chart(self.data_points_2d_actual)
            self.actual_fig_flag = False

    def show_kmeans_settings(self):
        self.select_num_clusters.set_visible(True) 
        self.dbscan_eps_min_samples.set_visible(False)
        self.dbscan_eps.set_visible(False)
        self.grid_stack_handler.add_chart(self.elbow_chart)
            
    def show_dbscan_settings(self):
        self.select_num_clusters.set_visible(False)
        self.dbscan_eps_min_samples.set_visible(True)  
        self.dbscan_eps.set_visible(True) 
        self.grid_stack_handler.remove_chart(self.elbow_chart.name)

    def show_agg_settings(self):
        self.select_num_clusters.set_visible(True)
        self.dbscan_eps_min_samples.set_visible(False)  
        self.dbscan_eps.set_visible(False) 
        self.grid_stack_handler.remove_chart(self.elbow_chart.name)
    
    def algo_settings_show(self,event):
        if self.select_algorithm.value == 'k-means':
            self.show_kmeans_settings()
        elif self.select_algorithm.value == 'DBscan':
            self.show_dbscan_settings()
        elif self.select_algorithm.value == 'Hierarchical Clustering':
            self.show_agg_settings()
    
    def bend_components_actions(self):
        self.run_clustering_button.param.watch(self.run_clustering,'value')
        self.select_algorithm.param.watch(self.algo_settings_show,'value')

    def get_clustered_data_io(self):
        sio = StringIO()
        try:
            self.reduction_dataset.to_csv(sio)
            sio.seek(0)
        except:
            pass
        return sio

    def get_trigger_button(self):
        return self.update_results_button
    def get_cluster_column(self):
        try:
            return self.output_dataset['Cluster']
        except:
            return None
            