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
import copy
from panel.layout.gridstack import GridStack
import logging
from .data_handler import *
from .geo_data_handler import * 
from .styles import *
from .clustering_module import *
from .regression_module import *
from .sankey_handler import *
from .gridstack_handler import grid_stack
from .classification_module import *
from .adv_panes import multiselect_input,select_input
from .text_field_module import *
from .static_pages import *

logging.basicConfig( level=logging.ERROR,force=True)

pn.extension('floatpanel')
pn.extension('gridstack')
pn.extension('tabulator')
pn.extension('plotly')
pn.extension('texteditor')
pn.extension(notifications=True)
pn.extension(

     design='bootstrap', template='material' 
)



class map_dashboard:
    def __init__(self,dataset=None) -> None:
        self.dataset = dataset
        self.features=None
        self.geo_handler = geojson_handler()
        self.geo_data = {}
        self.filter_columns_widgets=[]
        self.filter_columns_names = []
        self.time_column = None
        self.location_column=None
        self.value_column = None
        self.latest_filters_values={}
        self.active_main_tab = -1
        self.geo_feild_value = None
        self.geo_data_name =None
        self.chart_column = None
        self.filters_reset_values = {}
        self.design = Native
        self.map_layers = []
        self.map_obj = None
        self.map_base_option = ['OpenStreetMap', 'Stamen Toner', 'CartoDB positron', 'Cartodb dark_matter', 'Stamen Watercolor' , 'Stamen Terrain']
        self.map_color_option = color_brewer_palettes
        self.home_page_active = False #False means no expermint yet
        self.controls_latest_values={}
        self.mapping_failure_count = 0
        self.latest_overlap_mapping = 100
        self.control_column_filters_count = 0
        self.skip_map_flag = False
        self.clustering_module = None
        self.regression_module = None
        self.classification_module = None
        self.sankey_handler = SankeyPlotter()
        self.text_manager = text_field_manger()
        self.create_widgets()

    def create_filters_columns(self,filters_features,add_controls=True):
        for feature in filters_features:
            unique_values= list(self.dataset[feature].unique())
            feature_selction = pn.widgets.MultiChoice(name= feature ,options=unique_values,  visible=False)
            self.filters_reset_values[feature] = unique_values
            self.filter_columns_widgets.append(feature_selction)
        if add_controls:
            self.add_filters_to_control()
    def pop_old_filters(self):
        last_index = len(self.column_controls_compoent)-1
        for i in range(self.control_column_filters_count):
            self.column_controls_compoent.pop(last_index)
            last_index-=1
    def add_filters_to_control(self):
        #self.pop_old_filters()
        for w in self.filter_columns_widgets:
            self.column_controls_compoent.append(w)
            self.control_column_filters_count +=1
        self.column_controls_compoent.append(self.button_row)
        self.control_column_filters_count +=1

    def show_filters_widgets(self):
        
        def change_options(event):
            self.dynamic_filters(event.obj.name)
        for w in range(len(self.filter_columns_widgets)):
            self.filter_columns_widgets[w].param.watch(change_options,'value')

        for w in self.filter_columns_widgets:
            w.visible=True

    def get_filters_values(self):
        d = {}

        for i in range(len(self.filter_columns_names)):
            d[self.filter_columns_names[i]] = self.filter_columns_widgets[i].value

        return d


    def about_page_values(self):
        about_page_obj = about_page()
        self.about_page_component =  about_page_obj.get_page()

    def create_home_page(self):
        home_page_obj = home_page()
        self.create_experiment_button_page,self.create_example_button_page = home_page_obj.get_buttons()
        self.home_page_component =  home_page_obj.get_page()
        
    def uploading_dataset_components(self):
        self.first_sentence = pn.pane.Markdown('##### **Step 1:** Upload data.<br />', styles={"font-size": "10px"})
        self.file_input = pn.widgets.FileInput(name= 'Upload Data', accept='.csv,.xlsx', design=self.design)
        self.next_choose_geo = pn.widgets.Button(name='Next', button_type='primary', disabled= True, design=self.design)
        
        self.upload_dataset_component = pn.Column(self.first_sentence,self.file_input,self.next_choose_geo, visible=True)
    

    def choosing_geodata(self):
        geo_maps_names = list(self.geo_handler.get_maps_names())
        geo_maps_names.append('Upload Geojson')
        
        self.second_sentence = pn.pane.Markdown('##### **Step 2:** Plot GeoJSON.<br />', styles={"font-size": "10px"})
        self.select_geo = pn.widgets.Select(name='Select Map', options=geo_maps_names, design=self.design)
        self.next_map_button = pn.widgets.Button(name='Next', button_type='primary', disabled= True, design=self.design)
        self.skip_map_button = pn.widgets.Button(name='Skip', button_type='primary', disabled= False, design=self.design)
        self.geojson_input = pn.widgets.FileInput(name='Upload GeoJSON', visible=False, accept='.geojson,.json', design=self.design)
        self.button_row_map = pn.Row(self.next_map_button,self.skip_map_button, design=self.design)
        self.choose_geo_component = pn.Column(self.second_sentence,self.select_geo,self.geojson_input,self.button_row_map, visible=False)

    def choosing_columns_fields(self):
        self.select_location_column_tooltip = select_input('Location Column: ','Choose the column from the dataset with location identifiers that maps with GeoJSON Mapping field')
        self.select_year_column_tooltip = select_input('Time Column: ','Choose a column for date or time filtering or choose "None" if not applicable')
        self.select_value_column_tooltip = select_input('Initial Value Column (Y-axis): ','Choose the initial column for the y-axis in charts and values on the map if exist.')
        self.select_geo_field_tooltip = select_input('Geo Mapping Field: ','Choose the field in the GeoJSON file that maps with the location column and have the location identifiers.')
        self.select_chart_x_tooltip = select_input('Initial Charts Column (X-axis): ','Choose the initial column for the x-axis in charts')
        self.select_filter_columns_tooltip = multiselect_input('Filter Columns: ','Choose columns for dataset filters in the dashboard(not date/time type).')
        
        #########
        self.third_sentence = pn.pane.Markdown('##### **Step 3:** Settings.<br />', styles={"font-size": "10px"})
        self.select_filter_columns = self.select_filter_columns_tooltip.get_core_item()
        self.select_location_column = self.select_location_column_tooltip.get_core_item()
        self.select_year_column = self.select_year_column_tooltip.get_core_item()
        self.select_value_column = self.select_value_column_tooltip.get_core_item()
        self.select_geo_field = self.select_geo_field_tooltip.get_core_item()
        self.select_chart_x = self.select_chart_x_tooltip.get_core_item()
        self.create_map_final_button = pn.widgets.Button(name='Create Dashboard', button_type='primary', design=self.design)

        self.column_field_selection_compoent = pn.Column(self.third_sentence, self.select_filter_columns_tooltip.get_item(),self.select_location_column_tooltip.get_item(),
                                                         self.select_year_column_tooltip.get_item(),self.select_value_column_tooltip.get_item(),self.select_chart_x_tooltip.get_item(),
                                                         self.select_geo_field_tooltip.get_item(),self.create_map_final_button, visible=False)
        
    def creating_charts_controls_toggle(self):
        #Remove this button
        o = {"description": "Map chart"}
        self.setting_charts_show = pn.widgets.Toggle(button_type='light', button_style='solid', icon='settings-2', align='center', icon_size='16px')
        self.map_show = pn.widgets.Button(button_type='primary',description='Map Diagram', button_style='solid', icon='map-2', align='center', icon_size='16px',value=True)
        self.radar_show = pn.widgets.Button(button_type='primary',description='Radar Chart', button_style='outline', icon='chart-radar', align='center', icon_size='16px')
        self.heatmap_show = pn.widgets.Button(button_type='primary',description='Heatmap', button_style='outline', icon='grid-4x4', align='center', icon_size='16px')
        self.line_show = pn.widgets.Button(button_type='primary',description='Line Chart', button_style='outline', icon='chart-line', align='center', icon_size='16px')
        self.bar_show = pn.widgets.Button(button_type='primary',description='Bar Chart', button_style='outline', icon='chart-bar', align='center', icon_size='16px')
        self.box_show = pn.widgets.Button(button_type='primary',description='Box Chart', button_style='outline', icon='chart-candle-filled', align='center', icon_size='16px')
        self.violin_show = pn.widgets.Button(button_type='primary',description='Violin Chart', button_style='outline', icon='brand-drupal', align='center', icon_size='16px')        
        self.scatter_show = pn.widgets.Button(button_type='primary',description='Scatter Plot', button_style='outline', icon='grain', align='center', icon_size='16px')
        self.pie_show = pn.widgets.Button(button_type='primary',description='Pie Chart', button_style='outline', icon='chart-pie-2', align='center', icon_size='16px')
        self.sankey_show = pn.widgets.Button(button_type='primary',description='Sankey Diagram', button_style='outline', icon='chart-sankey', align='center', icon_size='16px')
        self.text_field_add_show = pn.widgets.Button(button_type='primary',description='Add New Text Field', button_style='solid', icon='text-plus', align='center', icon_size='16px')
        self.text_fields_column = self.text_manager.get_buttons_column()
        self.charts_show_control = pn.Column(self.map_show,self.bar_show,self.line_show,self.box_show,self.violin_show,self.scatter_show,self.pie_show,self.radar_show,self.heatmap_show,self.sankey_show,self.text_field_add_show,self.text_fields_column)
        self.charts_control = pn.WidgetBox(self.charts_show_control,name= 'charts',width=45, height=3050,styles={ "background":"#FAFAFA"})
        
    def creating_map_settings_controls(self):
        self.select_base_map = pn.widgets.Select(name='Base Map: ', options=self.map_base_option)
        self.select_color_map = pn.widgets.Select(name='Map Colouring: ', options=list(self.map_color_option.keys()),value = 'Red-Yellow-Blue')
        self.transparency_map_range = pn.widgets.IntSlider(name='Transparency Level: ',start=0, end=100, value=50, step=1, design=self.design)
        self.select_tooltip = pn.widgets.MultiChoice(name='Tooltip Columns: ', options=[])
        self.map_settings_card = pn.Card(self.select_base_map,self.select_color_map,self.transparency_map_range,self.select_tooltip, title="<h1 style='font-size: 15px;'>Map Settings</h1>", styles={"border": "none", "box-shadow": "none"})
    
    def creating_general_controls(self):
        self.final_sentence = pn.pane.Markdown('##### **Step 4:** Visualizations.<br />', styles={"font-size": "10px"})
        self.agg_buttons = pn.widgets.ToggleGroup(name='Aggregation Type', value='sum', options=['sum', 'min' , 'max' , 'mean'], behavior="radio",  design=self.design)
        self.year_range = pn.widgets.IntRangeSlider(name='Year',start=1997, end=2017, value=(1997, 2017), step=1, styles=custom_style, stylesheets=[stylesheet], design=self.design, visible=False)
        self.update_map_button = pn.widgets.Button(name='Update Dashboard', button_type='primary', design=self.design)
        self.reset_filters_button = pn.widgets.Button(name='Reset Filters', button_type='primary', design=self.design)
        self.freeze_show = pn.widgets.Button(button_type='primary',description='Freeze/Unfreeze the diagrams in the dashboard', button_style='outline', icon='snowflake', align='center', icon_size='14px')

        self.button_row = pn.Row(self.update_map_button,self.reset_filters_button,self.freeze_show, design=self.design)
    
    def creating_axes_controls(self):
        self.select_value_column_update_tooltip = select_input('Value Column (Y-axis): ','Choose the initial column for the y-axis in charts and values on the map if exist.')
        self.select_chart_x_update_tooltip = select_input('Charts Field (X-axis): ','Choose the initial column for the y-axis in charts and values on the map if exist.')
        self.select_legend_update_tooltip = select_input('Legend: ','Choose the initial column for the y-axis in charts and values on the map if exist.')
        

        self.select_value_column_update = self.select_value_column_update_tooltip.get_core_item()
        self.select_chart_x_update =  self.select_chart_x_update_tooltip.get_core_item()
        self.select_legend_update =  self.select_legend_update_tooltip.get_core_item()
        self.axes_settings_card = pn.Card(self.select_value_column_update_tooltip.get_item(),self.select_chart_x_update_tooltip.get_item(),self.select_legend_update_tooltip.get_item(), title="<h1 style='font-size: 15px;'>Axes Settings</h1>", styles={"border": "none", "box-shadow": "none"})
    
    def special_charts_settings(self):
        self.select_heatmap_fields_tooltip = multiselect_input('Heatmap Columns: ','Choose the initial column for the y-axis in charts and values on the map if exist.')
        self.select_sankey_source_tooltip = select_input('Sankey Diagram Source: ','Choose the initial column for the y-axis in charts and values on the map if exist.')
        self.select_sankey_target_tooltip = select_input('Sankey Diagram Target: ','Choose the initial column for the y-axis in charts and values on the map if exist.')

        self.select_heatmap_fields =  self.select_heatmap_fields_tooltip.get_core_item()
        self.select_sankey_source =  self.select_sankey_source_tooltip.get_core_item()
        self.select_sankey_target =  self.select_sankey_target_tooltip.get_core_item()
        self.special_charts_settings_card = pn.Card(self.select_heatmap_fields_tooltip.get_item(), self.select_sankey_source_tooltip.get_item(),self.select_sankey_target_tooltip.get_item(), title="<h1 style='font-size: 15px;'>Special Charts Settings</h1>", styles={"border": "none", "box-shadow": "none"})
    
    def creating_analysis_panel(self):
        self.clustering_show = pn.widgets.Toggle(button_type='primary', button_style='outline', align='center',name = 'Clustering' )
        self.regression_show = pn.widgets.Toggle(button_type='primary', button_style='outline', name='Regression', align='center')
        self.classification_show = pn.widgets.Toggle(button_type='primary', button_style='outline', name='Classification', align='center')
        self.analysis_panel = pn.Row(self.clustering_show, self.regression_show, self.classification_show)
    
    def creating_dashboard_controls(self): 
        self.creating_charts_controls_toggle()
        self.creating_map_settings_controls()
        self.creating_general_controls()
        self.creating_axes_controls()
        self.creating_analysis_panel()
        self.special_charts_settings()
        
        self.regular_controls=pn.Column(self.analysis_panel,self.map_settings_card,self.axes_settings_card,self.special_charts_settings_card, self.year_range,self.agg_buttons,name='controls')
        self.controls_row = pn.Tabs(self.regular_controls,self.charts_control)
        self.column_controls_compoent = pn.Column(self.final_sentence,self.regular_controls, sizing_mode='stretch_height',visible=False)
        
    def creating_titlebar_buttons(self):
        self.about_button = pn.widgets.Button(name="About", button_type="primary", icon ='alert-circle',align=('end','center'),margin = 0)
        self.home_button = pn.widgets.Button(name="Home", button_type="primary", icon ='home-2',align=('end','center'),margin = 0)
        self.menu_button = pn.widgets.Button(name="", button_type="primary", icon ='menu-2',align='center', icon_size= '24px',visible = False, margin = 3)
        self.create_experiment_button = pn.widgets.Button(name='Get Started', button_type='primary',  align=('end','center'),icon ='lamp')
        self.create_example_button = pn.widgets.Button(name='Demo', button_type='primary',  align=('end','center'),icon ='device-tv',margin = 0)
        self.titlebar_buttons = pn.Row(self.create_experiment_button,self.create_example_button,self.home_button,self.about_button,align=('end','center'),margin = 0)
        

    def create_main_area_widgets(self):
        empty_map = folium.Map(location=(41,-99), zoom_start=0,width='100%', height='100%')
        self.responsive_map = pn.pane.plot.Folium(empty_map, height=500, width=800, visible=False, name='Map', design=self.design)
        self.bar_chart = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='Bar chart',  design=self.design, margin=2),scroll=False)
        self.radar_chart = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='Radar chart', design=self.design, margin=2),scroll=False)
        self.heatmap_chart = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='Heatmap chart', design=self.design, margin=2),scroll=False)
        self.line_chart = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='line chart', design=self.design, margin=2),scroll=False)
        self.box_chart = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='Box chart', design=self.design, margin=2),scroll=False)
        self.violin_chart = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='Violin chart', design=self.design, margin=2),scroll=False)
        self.scatter_chart = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='Scatter chart', design=self.design, margin=2),scroll=False)
        self.pie_chart = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='Pie chart',  design=self.design, margin=2),scroll=False)
        self.sankey_chart = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='Sankey chart', design=self.design, margin=2),scroll=False)
        self.responsive_row = self.get_grid_stack([self.responsive_map])
        self.dashboard = pn.Row(self.charts_control,self.responsive_row,name='Dashboard')
        self.main_tabs = pn.Tabs( height=1800)
    
    def get_grid_stack(self,charts_dict):
        self.grid_stack_handler = grid_stack(charts_dict)
        self.text_manager.set_grid_area_obj(self.grid_stack_handler)
        return self.grid_stack_handler.get_gridstack()

  
    def create_example_videos_page(self):

        example_page_obj = example_page()

        self.example_page =  example_page_obj.get_page()

    def create_general_widgets(self):
        self.loading = pn.indicators.LoadingSpinner(value=True, size=20, name='Loading...', visible=False, design=self.design)

    def plotly_charts(self,bar_chart,scatter_chart,box_chart,line_chart,radar_chart,heatmap_chart,pie_chart,violin_chart,sankey_chart):
        self.bar_chart.clear()
        self.bar_chart.append(pn.pane.Plotly(bar_chart,name='Bar chart',  design=self.design, margin=2)) 
        self.box_chart.clear()
        self.box_chart.append(pn.pane.Plotly(box_chart,name='Box chart', design=self.design, margin=2))
        self.violin_chart.clear()
        self.violin_chart.append(pn.pane.Plotly(violin_chart,name='Violin chart', design=self.design, margin=2))
        self.scatter_chart.clear()
        self.scatter_chart.append(pn.pane.Plotly(scatter_chart,name='Scatter chart', design=self.design, margin=2))
        self.line_chart.clear()
        self.line_chart.append(pn.pane.Plotly(line_chart,name='line chart', design=self.design, margin=2))
        self.radar_chart.clear()
        self.radar_chart.append(pn.pane.Plotly(radar_chart,name='Radar chart', design=self.design, margin=2))
        self.heatmap_chart.clear()
        self.heatmap_chart.append(pn.pane.Plotly(heatmap_chart,name='Heatmap chart', design=self.design, margin=2))
        self.pie_chart.clear()
        self.pie_chart.append(pn.pane.Plotly(pie_chart,name='Pie chart',  design=self.design, margin=2))
        self.sankey_chart.clear()
        self.sankey_chart.append(pn.pane.Plotly(sankey_chart,name='Sankey chart',  design=self.design, margin=2))

    def refresh_charts(self):
        bar_chart = self.bar_chart.objects[0]
        self.bar_chart.clear()
        self.bar_chart.append(pn.pane.Plotly(bar_chart,name='Bar chart',  design=self.design, margin=2)) 
        box_chart = self.box_chart.objects[0]
        self.box_chart.clear()
        self.box_chart.append(pn.pane.Plotly(box_chart,name='Box chart', design=self.design, margin=2))
        violin_chart = self.violin_chart.objects[0]
        self.violin_chart.clear()
        self.violin_chart.append(pn.pane.Plotly(violin_chart,name='Violin chart', design=self.design, margin=2))
        scatter_chart = self.scatter_chart.objects[0]
        self.scatter_chart.clear()
        self.scatter_chart.append(pn.pane.Plotly(scatter_chart,name='Scatter chart', design=self.design, margin=2))
        line_chart = self.line_chart.objects[0]
        self.line_chart.clear()
        self.line_chart.append(pn.pane.Plotly(line_chart,name='line chart', design=self.design, margin=2))
        radar_chart = self.radar_chart.objects[0]
        self.radar_chart.clear()
        self.radar_chart.append(pn.pane.Plotly(radar_chart,name='Radar chart', design=self.design, margin=2))
        heatmap_chart = self.heatmap_chart.objects[0]
        self.heatmap_chart.clear()
        self.heatmap_chart.append(pn.pane.Plotly(heatmap_chart,name='Heatmap chart', design=self.design, margin=2))
        pie_chart = self.pie_chart.objects[0]
        self.pie_chart.clear()
        self.pie_chart.append(pn.pane.Plotly(pie_chart,name='Pie chart',  design=self.design, margin=2))
        sankey_chart = self.sankey_chart.objects[0]
        self.sankey_chart.clear()
        self.sankey_chart.append(pn.pane.Plotly(sankey_chart,name='Sankey chart',  design=self.design, margin=2))
        

    def create_widgets(self):
        #Creating widget columns for the control sidebar
        self.create_home_page()
        self.uploading_dataset_components()
        self.choosing_geodata()
        self.choosing_columns_fields()
        self.creating_dashboard_controls()
        #Create about page
        self.about_page_values()

        # Map page widgets
        self.create_main_area_widgets()
        
        # general_widgets
        self.create_general_widgets()

        #title_bar
        self.creating_titlebar_buttons()


        self.create_example_videos_page()
        # Highlevel widgets
        self.controls = pn.WidgetBox(self.upload_dataset_component, self.choose_geo_component, 
                                     self.column_field_selection_compoent, self.column_controls_compoent,
                                     self.loading,height=1100, width=330,styles={ "background":"#FAFAFA"},visible=False)
        
        self.map_area = pn.Column(self.main_tabs, sizing_mode='stretch_height',visible = False)

        
        if self.dataset!=None:
            self.update_widgets_map_create()
        
    def add_main_tab(self,obj):
        self.main_tabs.append(obj)
        self.active_main_tab+=1
        self.main_tabs.active = self.active_main_tab
        
    def remove_main_tap(self,obj):
        self.active_main_tab-=1
        self.main_tabs.active = self.active_main_tab
        self.main_tabs.remove(obj)

        

    def update_widgets_map_create(self):
        
        self.show_filters_widgets()
        
        if self.time_column:
            self.year_range.visible = True
            mn_year= int(min(list(self.dataset[self.time_column])))
            mx_year= int(max(list(self.dataset[self.time_column])))
            self.year_range.start= mn_year
            self.year_range.end= mx_year
            self.year_range.value = (mn_year,mx_year)
        
        self.loading.visible = False
        self.column_controls_compoent.visible = True
        if not self.skip_map_flag:
            self.responsive_map.visible = True
        self.responsive_row.visible = True

        columns = list(self.dataset.columns)
        columns_none = columns.copy()
        columns_none.append('None')
        numric_columns = list(self.dataset.select_dtypes(include=['number']).columns)

        self.select_value_column_update.options = numric_columns
        self.select_chart_x_update.options = columns
        self.select_legend_update.options = columns
        self.select_tooltip.options = columns
        self.select_heatmap_fields.options = numric_columns
        self.select_sankey_target.options = columns_none
        self.select_sankey_source.options = columns_none

        self.select_chart_x_update.value = self.chart_column
        self.select_value_column_update.value = self.value_column
        self.select_legend_update.value = self.chart_column
        self.select_sankey_target.value = columns_none[-1]
        self.select_sankey_source.value = columns_none[-1]
        
        self.add_main_tab(self.dashboard)
        
        #uploading data widget disable
        self.choose_geo_component.visible = False

        #columns chosing disable
        self.column_field_selection_compoent.visible = False
        
        
        self.controls.height=3085

    def update_widgets_dataset_columns_selection(self):
        #columns chosing enable and configure
        self.loading.visible = False

        columns = list(self.dataset.columns)
        columns_none = columns.copy()
        columns_none.append('None')
        numric_columns = list(self.dataset.select_dtypes(include=['number']).columns)
        if not self.skip_map_flag:
            geo_fields = self.geo_handler.get_all_fields()
            value_geo_field = 'GEOID' if 'GEOID' in geo_fields else geo_fields[0]
            self.select_geo_field.options = geo_fields
            self.select_geo_field.value = value_geo_field
            self.select_location_column.options = columns
            self.select_location_column.value = columns[0]


        self.column_field_selection_compoent.visible = True
        self.select_filter_columns.options = columns
        self.select_value_column.options = numric_columns
        self.select_year_column.options = columns_none
        self.select_chart_x.options = columns
        
        self.select_value_column.value = columns[0]
        self.select_year_column.value = columns_none[-1]
        self.select_chart_x.value = columns[0]
        self.select_filter_columns.size = min(18,len(columns))
        #uploading data widget disable
        self.choose_geo_component.visible = False    

    def show_geo_data_collection(self):
        self.loading.visible = False
        self.upload_dataset_component.visible= False
        self.choose_geo_component.visible = True
        
    def show_experiment_page(self,event):
        self.about_page_component.visible = False
        self.home_page_component.visible = False
        self.map_area.visible = True
        self.controls.visible = True
        self.menu_button.visible = True
        self.home_page_active = True
        self.create_experiment_button.visible = False
        self.example_page.visible= False
    def show_main_page(self):
        if self.home_page_active:
            self.map_area.visible = True
            self.controls.visible = True
        else:
            self.home_page_component.visible = True
        self.about_page_component.visible = False
        self.menu_button.disabled = False
        self.example_page.visible= False
    def show_about_page(self):
        self.map_area.visible = False
        self.controls.visible = False
        self.home_page_component.visible = False
        self.about_page_component.visible = True
        self.menu_button.disabled = True
        self.example_page.visible= False
    def show_home_page(self):
        self.map_area.visible = False
        self.controls.visible = False 
        self.about_page_component.visible = False
        self.home_page_component.visible = True
        self.menu_button.disabled = False
        self.example_page.visible= False
        self.create_experiment_button.visible = True
    def show_video_example(self,event):
        self.example_page.visible= True
        self.about_page_component.visible = False
        self.home_page_component.visible = False
        self.map_area.visible = False
        self.controls.visible = False
        self.menu_button.visible = False
        self.home_page_active = False
        
    def create_filtered_data_chart(self,features=None,agg=None,year_range=(1997, 2017)):
        if isinstance(self.dataset,pd.DataFrame):
            data_feature_filter = self.dataset.copy()
            
            for feature in features:
                if features[feature]==[]:
                    continue
                data_feature_filter=data_feature_filter[data_feature_filter[feature].isin(features[feature])]
            if self.time_column: 
                data_feature_filter=data_feature_filter[(data_feature_filter[self.time_column]>=year_range[0])&(data_feature_filter[self.time_column]<=year_range[1])]
            data_feature_filter_row = data_feature_filter
            data_feature_filter = data_feature_filter.astype({self.chart_column:'string'})
            if not self.legend_column or self.legend_column == self.chart_column:
                final_data = data_feature_filter.groupby([self.chart_column])[self.value_column].agg(agg).reset_index()
            else:
                final_data = data_feature_filter.groupby([self.chart_column,self.legend_column])[self.value_column].agg(agg).reset_index()
            if len(self.select_heatmap_fields.value)>0:
                final_data_heatmap = data_feature_filter.groupby(self.chart_column).agg({feature:agg for feature in self.select_heatmap_fields.value})
            else:
                final_data_heatmap = None

            return final_data,data_feature_filter,final_data_heatmap,data_feature_filter_row
        return None,None,None,None


    def create_charts(self,filtered_data,row_filtered_data,filtered_data_heatmap):
        fig_bar = None
        fig_pie = None
        fig_scatter = None
        cud_palette = ['#0072B2','#E69F00', '#56B4E9', '#009E73', '#F0E442' , '#CC79A7','#D55E00'
                       ,'#00429d', '#882255', '#44aa99', '#cc6677', '#332288', 
                        '#ddcc77', '#117733', '#88ccee', '#aa4499', '#661100', 
                        '#6699cc', '#aa4499', '#44aa99', '#ee6677', '#aa8822', 
                        '#882255', '#6699cc', '#4477aa', '#332288', '#88ccee', 
                        '#44aa99', '#117733', '#ddcc77', '#ee6677', '#99cc66', 
                        '#66ccee', '#aa4499', '#332288', '#88ccee', '#117733', 
                        '#ddcc77', '#cc6677', '#44aa99', '#88ccee', '#117733', 
                        '#88ccee', '#44aa99', '#117733', '#ddcc77', '#cc6677', 
                        '#4477aa', '#332288', '#88ccee', '#117733', '#ddcc77', 
                        '#ee6677', '#99cc66', '#66ccee', '#aa4499', '#332288', 
                        '#88ccee', '#44aa99', '#117733', '#ddcc77', '#ee6677', 
                        '#aa8822', '#882255', '#6699cc', '#4477aa', '#332288', 
                        '#88ccee', '#44aa99', '#117733', '#ddcc77', '#ee6677', 
                        '#aa4499', '#332288', '#88ccee', '#117733', '#ddcc77', 
                        '#cc6677', '#44aa99', '#88ccee', '#117733', '#88ccee', 
                        '#44aa99', '#117733', '#ddcc77', '#cc6677', '#4477aa', 
                        '#332288', '#88ccee', '#117733', '#ddcc77', '#ee6677', 
                        '#99cc66', '#66ccee', '#aa4499', '#332288', '#88ccee', 
                        '#44aa99', '#117733', '#ddcc77', '#ee6677', '#aa8822', 
                        '#882255', '#6699cc', '#4477aa']

        if isinstance(filtered_data,pd.DataFrame):
            color_column = None if self.chart_column == self.legend_column else self.legend_column
            fig_bar = px.histogram(filtered_data, x=self.chart_column, y=self.value_column, color=color_column, barmode="group", template="plotly_white", color_discrete_sequence=cud_palette).update_layout(margin=dict(l=20, r=20, t=5, b=5))
            fig_pie = px.pie(filtered_data, values=self.value_column, names=self.chart_column, template="plotly_white", color_discrete_sequence=cud_palette).update_layout(margin=dict(l=20, r=20, t=20, b=20))
            fig_scatter = px.scatter(filtered_data, x=self.chart_column, y=self.value_column, color=color_column, template="plotly_white", color_discrete_sequence=cud_palette).update_layout(margin=dict(l=20, r=20, t=5, b=5))
            fig_line = px.line(filtered_data, x=self.chart_column, y=self.value_column, color=color_column, template="plotly_white", color_discrete_sequence=cud_palette).update_layout(margin=dict(l=20, r=20, t=5, b=5))
            fig_box = px.box(row_filtered_data, x=self.chart_column, y=self.value_column, color=color_column, template="plotly_white", color_discrete_sequence=cud_palette).update_layout(margin=dict(l=20, r=20, t=5, b=5))
            fig_violin = px.violin(row_filtered_data, x=self.chart_column, y=self.value_column, color=color_column, template="plotly_white", color_discrete_sequence=cud_palette).update_layout(margin=dict(l=20, r=20, t=5, b=5))
            fig_radar = px.line_polar(filtered_data, theta=self.chart_column, r=self.value_column, color=color_column, line_close=True, template="plotly_white", color_discrete_sequence=cud_palette).update_layout(margin=dict(l=20, r=20, t=20, b=20))
            if  isinstance(filtered_data_heatmap,pd.DataFrame):
                fig_heatmap = px.imshow(filtered_data_heatmap,labels=dict(color="Value"),x=filtered_data_heatmap.columns,y=filtered_data_heatmap.index, text_auto=True, aspect="auto", template="plotly_white").update_layout(margin=dict(l=20, r=20, t=20, b=20),)
            else:
                fig_heatmap = go.Figure().update_layout(template="plotly_white")

            if self.select_sankey_target.value == 'None' or self.select_sankey_source.value == 'None':
                fig_sankey = go.Figure().update_layout(template="plotly_white")
            else:
                fig_sankey=self.sankey_handler.get_sankey_plot(self.curent_filter_data,self.select_sankey_source.value,self.select_sankey_target.value,self.value_column)
            fig_bar.layout.autosize = True
            fig_line.layout.autosize = True
            fig_pie.layout.autosize = True
            fig_box.layout.autosize = True
            fig_scatter.layout.autosize = True
            fig_radar.layout.autosize = True
            fig_heatmap.layout.autosize = True
            fig_violin.layout.autosize = True
            fig_sankey.layout.autosize = True
        self.plotly_charts(fig_bar,fig_scatter,fig_box,fig_line,fig_radar,fig_heatmap,fig_pie,fig_violin,fig_sankey)
        
    def add_columns_tooltip(self,final_data,cur_geo_data=None):
        geo_data = self.geo_data
        if cur_geo_data != None:
            geo_data = cur_geo_data
        temp_geo_data =  copy.deepcopy(geo_data)
        id_mapping = dict(zip(final_data[self.location_column], final_data[self.value_column]))
        values_dict = {}
        columns_list = [self.value_column]
        for value in self.select_tooltip.value:
            if value == self.value_column:
                continue
            values_dict[value] = dict(zip(final_data[self.location_column], final_data[value]))
            columns_list.append(value)
        for feature in temp_geo_data['features']:
            geoid = feature['properties'][self.geo_feild_value.split('.')[-1]]
            if geoid in id_mapping:
                feature['properties'][self.value_column] = id_mapping[geoid]
                for value in values_dict:
                    feature['properties'][value] = values_dict[value][geoid]

        return temp_geo_data,columns_list
    
    def create_filtered_data_map(self,features=None,agg=None,year_range=(1997, 2017)):
        data_feature_filter = self.dataset.copy()
        for feature in features:
            if features[feature]==[]:
                continue
            data_feature_filter=data_feature_filter[data_feature_filter[feature].isin(features[feature])]
        if self.time_column: 
            data_feature_filter=data_feature_filter[(data_feature_filter[self.time_column]>=year_range[0])&(data_feature_filter[self.time_column]<=year_range[1])]
        def get_unique_values(x):
            temp_set = set(x)
            temp_strs = [str(i) for i in temp_set]
            return ', '.join(temp_strs)
        aggregation_dict = {column : get_unique_values for column in self.select_tooltip.value if column != self.location_column }
        
        aggregation_dict[self.value_column] = agg
        

        final_data = data_feature_filter.groupby(self.location_column).agg(aggregation_dict).reset_index()
        return final_data
    
    def create_map(self,features=None,agg=None,year_range=(1997, 2017),transparency_level = 0.5, base_map = 'OpenStreetMap',color='YlGn'):
        base_map = 'OpenStreetMap' if base_map not in self.map_base_option else base_map
        m = folium.Map(location=self.geo_handler.get_center_point(), zoom_start=self.geo_handler.get_zoom_start(),width='100%', height='100%',tiles=base_map)
 
        if isinstance(self.dataset,pd.DataFrame) and features:
            final_data = self.create_filtered_data_map(features,agg,year_range)
            cur_geo_data,cur_zoom,cur_center_point = self.geo_handler.get_cur_map_specs(set(final_data[self.location_column]))
            m = folium.Map(location=cur_center_point, zoom_start=cur_zoom,width='100%', height='100%',tiles=base_map)
            temp_geo_data,tooltip_list = self.add_columns_tooltip(final_data,cur_geo_data)
            
            choropleth = folium.Choropleth(
                geo_data=cur_geo_data,
                name="choropleth"+str(len(self.map_layers)),
                data=final_data,
                columns=[self.location_column, self.value_column],
                key_on=self.geo_feild_value,
                fill_color=color,
                fill_opacity=transparency_level,
                nan_fill_color="White",
                nan_fill_opacity=0,
                line_opacity=0.2,
                nan_line_opacity=0,
                overlay=False
            ).add_to(m)

            style_function = lambda x: {'fillColor': '#ffffff', 
                                        'color':'#000000', 
                                        'fillOpacity': 0.1, 
                                        'weight': 0.1}
            highlight_function = lambda x: {'fillColor': '#000000', 
                                            'color':'#000000', 
                                            'fillOpacity': 0.50, 
                                            'weight': 0.1}
            map_info = folium.features.GeoJson(
                temp_geo_data,
                style_function=style_function, 
                control=False,
                highlight_function=highlight_function, 
                tooltip=folium.features.GeoJsonTooltip(
                    fields=tooltip_list,
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                )
            )
            m.add_child(map_info)
            m.keep_in_front(map_info)
        self.responsive_map.object=m
        

    def dynamic_filters(self,feature_changed):
        
        if len(self.filter_columns_widgets)<2:
            return None
        features = self.get_filters_values()
        year_range=self.year_range.value
        data_feature_filter = self.dataset.copy()
        change_flag = False
        empty_flag = True
        changed_filters = []
        for feature in features:
            if features[feature]!=self.latest_filters_values.get(feature,None):
                change_flag = True
                changed_filters.append(feature)
            
            if features[feature] != []:
                empty_flag=False
        
        self.latest_filters_values = features
        if not change_flag or empty_flag:
            return None
        
        for feature in features:
            if features[feature]==[]:
                continue
            data_feature_filter=data_feature_filter[data_feature_filter[feature].isin(features[feature])]
        if self.time_column: 
            data_feature_filter=data_feature_filter[(data_feature_filter[self.time_column]>=year_range[0])&(data_feature_filter[self.time_column]<=year_range[1])]
        
        for i in range(len(self.filter_columns_names)):
            
            if feature_changed== self.filter_columns_names[i] or len(self.filter_columns_widgets[i].value)!= 0:
                continue
            self.filter_columns_widgets[i].options = list(data_feature_filter[self.filter_columns_names[i]].unique())

    def reset_filters(self):
        for i in range(len(self.filter_columns_names)):
            self.filter_columns_widgets[i].options = self.filters_reset_values[self.filter_columns_names[i]]
            self.filter_columns_widgets[i].value = []
            self.latest_filters_values[self.filter_columns_names[i]] = self.filters_reset_values[self.filter_columns_names[i]]
    
    def add_dataset_tabs(self):
        pass
    

##########################################################################################

    def disable_map(self,event):
        self.map_settings_card.visible = False
        self.map_show.value = False
        self.map_show.visible = False
        self.select_geo_field_tooltip.set_visible(False)
        self.select_location_column_tooltip.set_visible(False)
        self.responsive_map.visible = False
        self.select_geo_field.visible = False
        self.select_location_column.visible = False
        self.skip_map_flag=True
        self.update_widgets_dataset_columns_selection()
    def show_control_side_bar(self,event):
        self.controls.visible = not self.controls.visible
    def geo_data_collecting(self,event):
        
        self.show_geo_data_collection()

    def about_page_handler(self,event):
        self.show_about_page()

    def home_page_handler(self,event):
        self.show_home_page()

    def dataset_input_handler(self,event):
        self.loading.visible = True
        data_value = self.file_input.value
        data_filename = self.file_input.filename
        if not data_value or not isinstance(data_value, bytes):
            return None
        
        dataset_preprocessor = data_handler(data_value,file_name=data_filename)
        self.dataset = dataset_preprocessor.get_data()

        self.add_main_tab(pn.pane.DataFrame(self.dataset,name='Dataset',max_width=1100,max_rows=20))
        self.next_choose_geo.disabled = False
        self.loading.visible = False

    def geojson_input_handler(self,event):
        geo_data_value = self.geojson_input.value
        if not geo_data_value or not isinstance(geo_data_value, bytes):
            return None
        
        geo_string_io = StringIO(geo_data_value.decode("utf8"))
        geojson_data = json.load(geo_string_io)
        self.next_map_button.disabled = False
        self.add_main_tab(pn.pane.JSON(geojson_data, name='GeoJSON',theme='light'))

    def reset_button_handler(self,event):
        self.reset_filters()

    def check_upload_geojson(self,event):
        if self.select_geo.value == 'Upload Geojson':
            self.geojson_input.visible=True
        else:
            self.next_map_button.disabled = False
            self.add_main_tab(pn.pane.JSON(self.geo_handler.get_single_geojson(self.select_geo.value), name='GeoJSON',theme='light'))

        
    def data_columns(self,event):
        self.loading.visible = True
        
        self.geo_data_name = self.select_geo.value
        if self.geo_data_name == 'Upload Geojson':
            self.geo_data_name= 'uploaded'
            geo_data_value = self.geojson_input.value
            if not geo_data_value or not isinstance(geo_data_value, bytes):
                self.geo_handler.add_geojson({}) 
            else:
                geo_string_io = StringIO(geo_data_value.decode("utf8"))
                geojson_data = json.load(geo_string_io)
                self.geo_handler.add_geojson(geojson_data)

        self.geo_handler.get_single_geojson(self.geo_data_name)
        self.update_widgets_dataset_columns_selection()

    def map_create_new(self,event):
        self.loading.visible = True
        data_value = self.file_input.value
        data_filename = self.file_input.filename
        self.filter_columns_names = list(self.select_filter_columns.value)
        

        
        dataset_preprocessor = data_handler(data_value,
                                            location_column=self.select_location_column.value,
                                            time_column=self.select_year_column.value,
                                            value_column= self.select_value_column.value,
                                            chart_column=self.select_chart_x.value,
                                            file_name=data_filename)
        
        self.dataset = dataset_preprocessor.get_data()
        self.curent_filter_data = self.dataset
        self.location_column = dataset_preprocessor.get_location_column()
        self.time_column = dataset_preprocessor.get_time_column()
        self.value_column = dataset_preprocessor.get_value_column()
        self.chart_column = dataset_preprocessor.get_chart_column()
        if not self.skip_map_flag:
            self.geo_feild_value = 'feature.properties.'+self.select_geo_field.value
            

            maping_overlap = self.geo_handler.check_overlap_locations(self.geo_data_name,set(self.dataset[self.location_column]),self.select_geo_field.value)
            if (maping_overlap<=99 and self.mapping_failure_count==0) or (maping_overlap<=99 and maping_overlap!=self.latest_overlap_mapping):
                self.loading.visible = False
                pn.state.notifications.error(f'<span style="font-family: sans-serif; font-size: 15px;">Location mapping overlap is {maping_overlap}%</span>',duration=0)
                self.mapping_failure_count+=1
                self.latest_overlap_mapping = maping_overlap
                return None
            
        
        self.create_filters_columns(self.filter_columns_names)
        
        if not self.skip_map_flag:
            self.geo_data = self.geo_handler.get_single_geojson(self.geo_data_name,set(self.dataset[self.location_column]),self.select_geo_field.value)
            
            self.create_map()

        self.update_widgets_map_create()


    def check_what_changed(self,filters_values):
        #all 0 , map only 1 , charts only 2
        change_map = False
        change_charts = False
        final_change = 0
        temp_controls_latest_values = {}
        map_values = ['transparency_map_range','select_color_map','select_base_map','select_tooltip']
        chart_values = ['select_legend_update' , 'select_chart_x_update','select_heatmap_fields','select_sankey_source','select_sankey_target' ]
        all_values = ['agg_buttons' , 'year_range', 'select_value_column_update']

        temp_controls_latest_values['agg_buttons'] = self.agg_buttons.value
        temp_controls_latest_values['year_range'] = self.year_range.value
        temp_controls_latest_values['select_value_column_update'] = self.select_value_column_update.value
        temp_controls_latest_values['transparency_map_range'] = self.transparency_map_range.value
        temp_controls_latest_values['select_color_map'] = self.select_color_map.value
        temp_controls_latest_values['select_base_map'] = self.select_base_map.value
        temp_controls_latest_values['select_tooltip'] = self.select_tooltip.value
        temp_controls_latest_values['select_chart_x_update'] = self.select_chart_x_update.value
        temp_controls_latest_values['select_legend_update'] = self.select_legend_update.value
        temp_controls_latest_values['select_heatmap_fields'] = self.select_heatmap_fields.value
        temp_controls_latest_values['select_sankey_source'] = self.select_sankey_source.value
        temp_controls_latest_values['select_sankey_target'] = self.select_sankey_target.value

        temp_controls_latest_values['filters_values'] = filters_values
        
        if len(self.controls_latest_values) >0:
            
            for value in temp_controls_latest_values:
                if self.controls_latest_values[value]!=temp_controls_latest_values[value]:
                    if value in map_values:
                        change_map = True
                    elif value in chart_values:
                        change_charts= True
                    else:
                        change_map = True
                        change_charts= True
            
            if change_map and not change_charts:
                final_change =  1
            elif not change_map and change_charts:
                final_change =  2
        self.controls_latest_values = temp_controls_latest_values
        return final_change
    def check_heatmap_selections(self):
        if len(self.select_heatmap_fields.value)>0:
            self.heatmap_show.disabled = False
        else:
            self.heatmap_show.disabled = True
            self.heatmap_show.value = False

    def check_sankey_selections(self):
        if self.select_sankey_target.value == 'None' or self.select_sankey_source.value == 'None':
            self.sankey_show.disabled=True
            self.sankey_show.value = False
        else:
            self.sankey_show.disabled = False
    def map_update(self,event):
        new_features = self.get_filters_values()
        change_type = 2
        if not self.skip_map_flag:
            change_type = self.check_what_changed(new_features)
        #Change both
        agg = self.agg_buttons.value
        year_range_value=self.year_range.value
        self.value_column = self.select_value_column_update.value
        #Change map only
        transparency_level = self.transparency_map_range.value/100
        map_coloring = self.map_color_option[self.select_color_map.value]
        base_map = self.select_base_map.value
        #Change charts only
        self.chart_column = self.select_chart_x_update.value
        self.legend_column = self.select_legend_update.value

        filtered_data,row_filtered_data,filtered_data_heatmap,self.curent_filter_data = self.create_filtered_data_chart(new_features,agg,year_range_value)
        self.update_other_components()
        if change_type == 0:
            thread1 = threading.Thread(target=self.create_map, args=(new_features,agg,year_range_value,transparency_level,base_map,map_coloring,))
            thread2 = threading.Thread(target=self.create_charts, args=(filtered_data,row_filtered_data,filtered_data_heatmap,))

            # Start both threads
            thread1.start()
            thread2.start()

            # Wait for both threads to finish
            thread1.join()
            thread2.join()

        elif change_type == 1:
            self.create_map(new_features,agg,year_range_value,transparency_level,base_map,map_coloring)
        
        else:
            self.create_charts(filtered_data,row_filtered_data,filtered_data_heatmap)
        self.check_heatmap_selections()
        self.check_sankey_selections()

    def update_other_components(self):
        if self.clustering_module != None:
            self.clustering_module.set_dataset(self.curent_filter_data)
        if self.regression_module != None:
            self.regression_module.set_dataset(self.curent_filter_data)
        if self.classification_module != None:
            self.classification_module.set_dataset(self.curent_filter_data)

    def add_clustering_results(self,event):
        #try:
            clusters = self.clustering_module.get_cluster_column()
            clusters = clusters.astype(int)
            self.dataset['Clusters'] = clusters
            if 'Clusters' not in self.select_value_column_update.options:
                self.select_value_column_update.options.append('Clusters')
                #self.select_heatmap_fields.options.append('Clusters')
                self.axes_settings_card.clear()
                self.special_charts_settings_card.clear()
                self.axes_settings_card.objects = [self.select_value_column_update_tooltip.get_item(),self.select_chart_x_update_tooltip.get_item(),self.select_legend_update_tooltip.get_item()]
                self.special_charts_settings_card.objects = [self.select_heatmap_fields_tooltip.get_item(), self.select_sankey_source_tooltip.get_item(),self.select_sankey_target_tooltip.get_item()]
    def add_regression_results(self,event):
        #try:
            regression_results = self.regression_module.get_regression_column()
            self.dataset['Regression'] = regression_results
            if 'Regression' not in self.select_value_column_update.options:
                self.select_value_column_update.options.append('Regression')
                #self.select_heatmap_fields.options.append('Regression')
                self.axes_settings_card.clear()
                self.special_charts_settings_card.clear()
                self.axes_settings_card.objects = [self.select_value_column_update_tooltip.get_item(),self.select_chart_x_update_tooltip.get_item(),self.select_legend_update_tooltip.get_item()]
                self.special_charts_settings_card.objects = [self.select_heatmap_fields_tooltip.get_item(), self.select_sankey_source_tooltip.get_item(),self.select_sankey_target_tooltip.get_item()]

    def add_classification_results(self,event):
        #try:
            classification_results = self.classification_module.get_classification_column()
            self.dataset['Classification'] = classification_results
            if 'Classification' not in self.select_value_column_update.options:
                self.select_value_column_update.options.append('Classification')
                #self.select_heatmap_fields.options.append('Classification')
                self.axes_settings_card.clear()
                self.special_charts_settings_card.clear()
                self.axes_settings_card.objects = [self.select_value_column_update_tooltip.get_item(),self.select_chart_x_update_tooltip.get_item(),self.select_legend_update_tooltip.get_item()]
                self.special_charts_settings_card.objects = [self.select_heatmap_fields_tooltip.get_item(), self.select_sankey_source_tooltip.get_item(),self.select_sankey_target_tooltip.get_item()]

    def clustering_module_handeling(self,event):
        if self.clustering_show.value:
            if self.clustering_module ==None:
                self.clustering_module = clustering_module(self.curent_filter_data)
                self.clustering_main_area = self.clustering_module.get_main_area()
                self.clustering_controls = self.clustering_module.get_controls()
                self.regular_controls.insert(1,self.clustering_controls)
                self.clustering_trigger = self.clustering_module.get_trigger_button()
                self.clustering_trigger.param.watch(self.add_clustering_results,'value')
            else:
                self.clustering_main_area = self.clustering_module.get_main_area()
                self.clustering_controls.visible=True
            self.add_main_tab(self.clustering_main_area)
            self.grid_stack_handler.refresh_grid_stack()
            if self.regression_module!=None:
                self.regression_module.refresh_main_page()
            if self.classification_module!=None:
                self.classification_module.refresh_main_page()
        else:
            self.clustering_controls.visible=False
            self.remove_main_tap(self.clustering_main_area)
    
    def regression_module_handeling(self,event):
        if self.regression_show.value:
            if self.regression_module ==None:
                self.regression_module = regression_module(self.curent_filter_data)
                self.regression_main_area = self.regression_module.get_main_area()
                self.regression_controls = self.regression_module.get_controls()
                self.regular_controls.insert(1,self.regression_controls)
                self.regression_trigger = self.regression_module.get_trigger_button()
                self.regression_trigger.param.watch(self.add_regression_results,'value')
            else:
                self.regression_main_area = self.regression_module.get_main_area()
                self.regression_controls.visible=True
            self.add_main_tab(self.regression_main_area)
            self.grid_stack_handler.refresh_grid_stack()
            if self.clustering_module!=None:
                self.clustering_module.refresh_main_page()
            if self.classification_module!=None:
                self.classification_module.refresh_main_page()
        else:
            self.regression_controls.visible=False
            self.remove_main_tap(self.regression_main_area)
    
    def classification_module_handeling(self,event):
        if self.classification_show.value:
            if self.classification_module ==None:
                self.classification_module = classification_module(self.curent_filter_data)
                self.classification_main_area = self.classification_module.get_main_area()
                self.classification_controls = self.classification_module.get_controls()
                self.regular_controls.insert(1,self.classification_controls)
                self.classification_trigger = self.classification_module.get_trigger_button()
                self.classification_trigger.param.watch(self.add_classification_results,'value')
            else:
                self.classification_main_area = self.classification_module.get_main_area()
                self.classification_controls.visible=True
            self.add_main_tab(self.classification_main_area)
            self.grid_stack_handler.refresh_grid_stack()
            if self.clustering_module!=None:
                self.clustering_module.refresh_main_page()
            if self.regression_module!=None:
                self.regression_module.refresh_main_page()
        else:
            self.classification_controls.visible=False
            self.remove_main_tap(self.classification_main_area)
    def show_map_chart(self,event):
        if self.map_show.clicks%2==0:
            self.map_show.button_style='outline'
            value= False
        else:
            value = True
            self.map_show.button_style='solid'

        if value:
            self.grid_stack_handler.add_chart(self.responsive_map)
        else:
            self.grid_stack_handler.remove_chart(self.responsive_map.name)

    def show_bar_chart(self,event):
        print(self.bar_show.clicks,self.bar_show.clicks%2==0)
        if self.bar_show.clicks%2!=0:
            self.bar_show.button_style='outline'
            value= False
        else:
            value = True
            self.bar_show.button_style='solid'
        print(value)
        if value:
            self.grid_stack_handler.add_chart(self.bar_chart)
        else:
            self.grid_stack_handler.remove_chart(self.bar_chart.name)

    def show_scatter_chart(self,event):
        if self.scatter_show.clicks%2!=0:
            self.scatter_show.button_style='outline'
            value= False
        else:
            value = True
            self.scatter_show.button_style='solid'

        if value:
            self.grid_stack_handler.add_chart(self.scatter_chart)
        else:
            self.grid_stack_handler.remove_chart(self.scatter_chart.name)

    def show_box_chart(self,event):
        if self.box_show.clicks%2!=0:
            self.box_show.button_style='outline'
            value= False
        else:
            value = True
            self.box_show.button_style='solid'

        if value:
            self.grid_stack_handler.add_chart(self.box_chart)
        else:
            self.grid_stack_handler.remove_chart(self.box_chart.name)

    def show_pie_chart(self,event):
        if self.pie_show.clicks%2!=0:
            self.pie_show.button_style='outline'
            value= False
        else:
            value = True
            self.pie_show.button_style='solid'

        if value:
            self.grid_stack_handler.add_chart(self.pie_chart)
        else:
            self.grid_stack_handler.remove_chart(self.pie_chart.name)

    def show_line_chart(self,event):
        if self.line_show.clicks%2!=0:
            self.line_show.button_style='outline'
            value= False
        else:
            value = True
            self.line_show.button_style='solid'

        if value:
            self.grid_stack_handler.add_chart(self.line_chart)
        else:
            self.grid_stack_handler.remove_chart(self.line_chart.name)

    def show_radar_chart(self,event):
        if self.radar_show.clicks%2!=0:
            self.radar_show.button_style='outline'
            value= False
        else:
            value = True
            self.radar_show.button_style='solid'

        if value:
            self.grid_stack_handler.add_chart(self.radar_chart)
        else:
            self.grid_stack_handler.remove_chart(self.radar_chart.name)

    def show_heatmap_chart(self,event):
        if self.heatmap_show.clicks%2!=0:
            self.heatmap_show.button_style='outline'
            value= False
        else:
            value = True
            self.heatmap_show.button_style='solid'

        if value:
            self.grid_stack_handler.add_chart(self.heatmap_chart)
        else:
            self.grid_stack_handler.remove_chart(self.heatmap_chart.name)

    def show_sankey_chart(self,event):
        if self.sankey_show.clicks%2!=0:
            self.sankey_show.button_style='outline'
            value= False
        else:
            value = True
            self.sankey_show.button_style='solid'

        if value:
            self.grid_stack_handler.add_chart(self.sankey_chart)
        else:
            self.grid_stack_handler.remove_chart(self.sankey_chart.name)

    def show_violin_chart(self,event):
        if self.violin_show.clicks%2!=0:
            self.violin_show.button_style='outline'
            value= False
        else:
            value = True
            self.violin_show.button_style='solid'

        if value:
            self.grid_stack_handler.add_chart(self.violin_chart)
        else:
            self.grid_stack_handler.remove_chart(self.violin_chart.name)

    def freeze_handler(self,event):
        if self.freeze_show.clicks%2!=0:
            self.freeze_show.button_style='outline'
            value= False
        else:
            value = True
            self.freeze_show.button_style='solid'

        dynamic_flag = not value
        self.grid_stack_handler.dynamic(dynamic_flag)


    def add_new_text(self,event):
        self.text_manager.create_new_text_field()
##########################################################################################
    def bend_components_actions(self):
        self.geojson_input.param.watch(self.geojson_input_handler,'value')
        self.file_input.param.watch(self.dataset_input_handler,'value')
        self.next_map_button.param.watch(self.data_columns,'value')
        self.create_map_final_button.param.watch(self.map_create_new,'value')
        self.update_map_button.param.watch(self.map_update, 'value')
        self.select_geo.param.watch(self.check_upload_geojson, 'value')
        self.reset_filters_button.param.watch(self.reset_button_handler, 'value')
        self.about_button.param.watch(self.about_page_handler, 'value')
        self.home_button.param.watch(self.home_page_handler, 'value')
        self.next_choose_geo.param.watch(self.geo_data_collecting,'value')
        self.menu_button.param.watch(self.show_control_side_bar,'value')
        self.create_experiment_button.param.watch(self.show_experiment_page,'value')
        self.create_experiment_button_page.param.watch(self.show_experiment_page,'value')
        self.skip_map_button.param.watch(self.disable_map,'value')
        self.create_example_button.param.watch(self.show_video_example,'value') 
        self.create_example_button_page.param.watch(self.show_video_example,'value') 

        self.map_show.param.watch(self.show_map_chart,'value')
        self.line_show.param.watch(self.show_line_chart,'value') 
        self.bar_show.param.watch(self.show_bar_chart,'value') 
        self.box_show.param.watch(self.show_box_chart,'value') 
        self.scatter_show.param.watch(self.show_scatter_chart,'value') 
        self.pie_show.param.watch(self.show_pie_chart,'value') 
        self.radar_show.param.watch(self.show_radar_chart,'value')
        self.heatmap_show.param.watch(self.show_heatmap_chart,'value')
        self.violin_show.param.watch(self.show_violin_chart,'value')
        self.sankey_show.param.watch(self.show_sankey_chart,'value')
        self.freeze_show.param.watch(self.freeze_handler,'value')
        self.text_field_add_show.param.watch(self.add_new_text,'value')

        self.clustering_show.param.watch(self.clustering_module_handeling,'value')
        self.classification_show.param.watch(self.classification_module_handeling,'value')
        self.regression_show.param.watch(self.regression_module_handeling,'value')
    def create_template(self):
        self.bend_components_actions()

        title = pn.pane.Markdown("", styles={"font-size": "18px", "font-weight": "bold", "color":"White"}, sizing_mode="stretch_width") 
        logo = pn.pane.SVG("/code/map_app/GUI/Static_data/lamba_logo.svg",align=('end', 'end'), height=60,margin=1) 
        title_bar = pn.Row(self.menu_button,
                        logo,
                        title, 
                        self.titlebar_buttons,
                        styles={"align":"center", "background":"#0172B6",  "width_policy":"max",   "sizing_mode":"stretch_width"}
                        , design=self.design, sizing_mode="stretch_width")
        
        
        self.app_layout = pn.Column(
            title_bar,
            self.home_page_component,
            self.about_page_component,
            self.example_page,
            pn.Row(
                self.controls,
                self.map_area,
                sizing_mode='stretch_width', design=self.design
            ),
            sizing_mode='stretch_both', design=self.design,
        )
        return self.app_layout

        
    

   

        
def createApp():
    md = map_dashboard()

    return md.create_template().servable()