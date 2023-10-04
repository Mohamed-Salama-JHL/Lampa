#Not used

import numpy as np
import param
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
import panel as pn
import numpy as np
import hvplot.pandas  # noqa
import hvplot.dask
import pandas as pd
import folium
import panel as pn
import hvplot.pandas
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import CategoricalColorMapper,CheckboxGroup, CustomJS
#import matplotlib.pyplot as plt
from urllib.request import urlopen
import json
import plotly.express as px

    
class map_dashboard:
    def __init__(self,data_per_county) -> None:
        self.data_per_county = data_per_county
        self.prepare_geometry_data()
        self.create_widgets()

    def prepare_geometry_data(self):
        d = {}
        with urlopen('https://gist.githubusercontent.com/sdwfrost/d1c73f91dd9d175998ed166eb216994a/raw/e89c35f308cee7e2e5a784e1d3afc5d449e9e4bb/counties.geojson') as response:
            self.counties = json.load(response)

        FIPS = list(self.data_per_county['FIPS'].unique())
        new_geo_list=[]

        for county in self.counties['features']:
            if county['properties']['GEOID'] in FIPS:
                d[county['properties']['GEOID']]=True
                new_geo={
                    'GEOID':county['properties']['GEOID'],
                    'geometry':county['geometry'],
                    'id':county['properties']['GEOID']
                }
                #new_geo['geometry']['id']=county['properties']['GEOID']
                
                new_geo_list.append(county)
                
        self.counties['features']=new_geo_list
    
    def create_widgets(self):
        self.categories = list(self.data_per_county['Categories'].unique())
        self.features = list(self.data_per_county['Feature'].unique())

        self.multi_select_categories = pn.widgets.MultiSelect(name='Categories', value=self.categories,options=self.categories, size=12)

        self.multi_select_features = pn.widgets.MultiSelect(name='Features', value=self.features,options=self.features, size=18)

        self.agg_buttons = pn.widgets.ToggleGroup(name='Aggregation type', value='sum', options=['sum', 'min' , 'max' , 'mean'], behavior="radio")

        self.year_range = pn.widgets.IntRangeSlider(name='Year',start=1997, end=2017, value=(1997, 2017), step=5)

        self.button = pn.widgets.Button(name='Map update', button_type='primary')


    def create_map(self,features,agg,year_range=(1997, 2017),color='YlGn'):
        data_year_filter=self.data_per_county[(self.data_per_county.Year>=year_range[0])&(self.data_per_county.Year<=year_range[1])]
        data_feature_filter=data_year_filter[data_year_filter.Feature.isin(features)]
        final_data = data_feature_filter.groupby(['County_state_formated','State','County','FIPS']).value.agg(agg).reset_index()

        m = folium.Map(location=[41, -99], zoom_start=5,width='100%', height='100%')

        choropleth = folium.Choropleth(
            geo_data=self.counties,
            name="choropleth",
            data=final_data,
            columns=["FIPS", "value"],
            key_on='feature.properties.GEOID',
            fill_color="RdYlBu",
            fill_opacity=0.5,
            nan_fill_color="White",
            nan_fill_opacity=0,
            line_opacity=0.2,
            nan_line_opacity=0
        ).add_to(m)
        return m 
    
    def create_template(self):
        fig = self.create_map(self.features,self.agg_buttons.value)

        responsive = pn.pane.plot.Folium(fig, height=1000, width=1000)

        def change(event):
            new_features = list(self.multi_select_features.value)
            agg = self.agg_buttons.value
            year_range_value=self.year_range.value
            fig = self.create_map(new_features,agg,year_range_value)
            responsive.object=fig

        self.button.param.watch(change, 'value')



        controls = pn.WidgetBox(self.year_range,self.multi_select_features,self.agg_buttons,self.button, sizing_mode='stretch_height')
        map_area = pn.WidgetBox(responsive, sizing_mode='stretch_height')
        sidebar = pn.FloatPanel(controls, title='Controls', sizing_mode='fixed', width=250)

        # Arrange the FloatPanel and the map area using a GridSpec layout
        title = pn.pane.Markdown("# My Dashboard", sizing_mode="stretch_width", style={"font-size": "24px", "font-weight": "bold"})
        title_bar = pn.Row(title, sizing_mode="stretch_width", background="#f0f0f0", align="center")

        config = {"headerControls": {
        "close": "remove",
        'maximize': "remove",
        'normalize': "remove",
        'minimize': "remove",
        }}
        
        floatmap = pn.layout.FloatPanel(map_area, name='Map', margin=20, config=config)
        floatcontrols = pn.layout.FloatPanel(controls, name='Controls', margin=20, config=config,theme='none', sizing_mode='stretch_height')


        app_layout = pn.Column(
            title_bar,
            pn.Row(
                controls,
                map_area,
                sizing_mode='stretch_width',
            ),
            sizing_mode='stretch_both',
        )
        return app_layout

        
        
        
def createApp():
    data_per_county_pivot= pd.read_csv(r"C:\James Hutton\Syngenta project\Data\data_per_county_pivot_fips.csv",dtype={'FIPS':'string'})
    md = map_dashboard(data_per_county_pivot)

    return md.create_template().servable()