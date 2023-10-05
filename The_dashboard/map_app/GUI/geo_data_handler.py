import os
import json
import copy
import numpy as np
from .helper_functions import *
class geojson_handler:
    def __init__(self):
        self.folder_name = './map_app/GUI/geojsons'
        self.selected_map = None
        self.return_geojson= None
        self.geojson_dict = {}
        self.load_geojsons()
        
    def load_geojsons(self):
        self.geojson_dict['None']={}
        for filename in os.listdir(self.folder_name):
            if filename.endswith('.geojson'):
                file_path = os.path.join(self.folder_name, filename)
                with open(file_path, 'r') as file:
                    try:
                        geojson_data = json.load(file)
                        name = filename.replace('.geojson','')
                        self.geojson_dict[name] = geojson_data
                    except Exception as e:
                        print(f"Error loading {filename}: {str(e)}")
    def add_geojson(self,geo_data):
        self.geojson_dict['uploaded'] = geo_data
        self.selected_map = 'uploaded'

    def get_single_geojson(self,name,locations=None,geo_field='GEOID'):

        self.selected_map = name
        self.geo_field = geo_field
        self.return_geojson = self.geojson_dict[self.selected_map]
        if locations!=None:
            self.prepare_geometry_data(locations,geo_field)

        return self.return_geojson

    def get_geojsons(self):
        return self.geojson_dict
    
    def get_maps_names(self):
        return self.geojson_dict.keys()
    
    def get_center_point(self,geo_data=None):
        cur_geo_data = self.return_geojson
        if geo_data !=None:
            cur_geo_data = geo_data
            
        if not cur_geo_data:
            return 41, -99
        total_lat, total_lon = 0, 0
        num_points = 0

        for feature in cur_geo_data['features']:
            geometry = feature['geometry']
            if geometry['type'] == 'Polygon':
                coords = geometry['coordinates'][0]  # For Polygon, the coordinates are nested in another list

            for lon, lat in coords:
                total_lon += lon
                total_lat += lat
                num_points += 1

        center_lon = total_lon / num_points
        center_lat = total_lat / num_points

        return [center_lat, center_lon]
    
    # after choosing mapping
    def prepare_geometry_data(self,locations,geo_field='GEOID'):
        new_geo_list=[]
        
        for location in self.return_geojson['features']:
            if location['properties'][geo_field] in locations:
                new_geo_list.append(location)

        self.return_geojson['features']=new_geo_list

    def _get_largest_distance(self,geo_data):
        largest_distance = 0
        for feature in geo_data['features']:
            geometry = feature['geometry']
            if geometry['type'] == 'Polygon':
                coords = geometry['coordinates'][0]  
            elif geometry['type'] == 'MultiPolygon':
                coords = [c for part in geometry['coordinates'] for c in part[0]]  
            else:
                continue
            try:
                for i, (lon1, lat1) in enumerate(coords):
                    for (lon2, lat2) in coords[i + 1:]:
                        distance = haversine_distance(lon1, lat1, lon2, lat2)
                        if distance > largest_distance:
                            largest_distance = distance
            except:
                print(feature)

        return largest_distance


    def get_zoom_start(self,geo_data=None):
        cur_geo_data = self.return_geojson
        if geo_data !=None:
            cur_geo_data = geo_data
        if cur_geo_data==None:
            return 0
        actual_distance = self._get_largest_distance(cur_geo_data)
        
        point1 = (0.3, 17)
        point2 = (228, 5.2)
        point3 = (161, 7)
        point4 = (70, 7.5)

        coefficients = np.polyfit([point1[0], point2[0], point3[0], point4[0]],
                                [point1[1], point2[1], point3[1], point4[1]], 3)
        a, b, c, d = coefficients

        output = a * actual_distance ** 3 + b * actual_distance ** 2 + c * actual_distance + d
        output = max(1, min(17, output))
    
        return int(output) 
    

    def get_all_fields(self):
        return get_final_value_keys(self.return_geojson)
    

    def get_cur_geojson(self,locations):
        new_geo_list=[]
        
        for location in self.return_geojson['features']:
            if location['properties'][self.geo_field] in locations:
                new_geo_list.append(location)

        geo_copy = copy.deepcopy(self.return_geojson)
        geo_copy['features']=new_geo_list

        return geo_copy

    def get_cur_map_specs(self,locations):
        cur_geojson = self.get_cur_geojson(locations)
        cur_zoom_lvl = self.get_zoom_start(cur_geojson)
        cur_center_poin = self.get_center_point(cur_geojson)
        return cur_geojson,cur_zoom_lvl,cur_center_poin

