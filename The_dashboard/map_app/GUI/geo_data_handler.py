import os
import json
import copy
import numpy as np
from .helper_functions import *
class geojson_handler:
    def __init__(self):
        self.folder_name = 'C:/James Hutton/Syngenta project/Lampa/The_dashboard/map_app/GUI/geojsons'
        #self.folder_name = './map_app/GUI/geojsons'
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
            coords = []
            if geometry['type'] == 'Polygon':
                coords = geometry['coordinates'][0]  
            elif geometry['type'] == 'MultiPolygon':
                for polygon_coords in geometry['coordinates']:
                    coords.extend(polygon_coords[0])
            for lon, lat in coords:
                total_lon += lon
                total_lat += lat
                num_points += 1
        if num_points>0:
            center_lon = total_lon / num_points
            center_lat = total_lat / num_points

            return [center_lat, center_lon]
        else:
            return 41, -99
    
    def check_overlap_locations(self,name, locations,geo_field='GEOID'):
        count = 0
        locations_count = len(locations)
        for location in self.geojson_dict[name]['features']:
            if location['properties'][geo_field] in locations:
                count+=1
        
        return (count/locations_count)*100
    
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

    def _get_largest_distance_2(self,geo_data):
        largest_distance = 0
        min_latitude = 100000000
        min_longitude = 100000000

        max_latitude = -100000000
        max_longitude = -100000000
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
                    min_latitude = min(min_latitude,lat1)
                    min_longitude = min(min_longitude,lon1)

                    max_latitude = max(max_latitude,lat1)
                    max_longitude = max(max_longitude,lon1)
            except:
                print(feature)
        largest_distance = haversine_distance(min_longitude, min_latitude, max_longitude, max_latitude)
        return largest_distance
    def get_zoom_start(self,geo_data=None):
        cur_geo_data = self.return_geojson
        if geo_data !=None:
            cur_geo_data = geo_data
        if cur_geo_data==None:
            return 0
        #return self.get_zoom_points(cur_geo_data)
        actual_distance = self._get_largest_distance_2(cur_geo_data)
        '''
        point1 = (0.3, 17)
        point2 = (228, 5.2)
        point3 = (161, 7)
        point4 = (70, 7.5)

        coefficients = np.polyfit([point1[0], point2[0], point3[0], point4[0]],
                                [point1[1], point2[1], point3[1], point4[1]], 3)
        a, b, c, d = coefficients

        output = a * actual_distance ** 3 + b * actual_distance ** 2 + c * actual_distance + d
        output = max(1, min(17, output))
        '''
        zoom_levels = {
            0: 156412,
            1: 78206,
            2: 39103,
            3: 19551,
            4: 9776,
            5: 4888,
            6: 2444,
            7: 1222,
            8: 611,
            9: 305,
            10: 152,
            11: 76,
            12: 38,
            13: 19,
            14: 10,
            15: 5,
            16: 2,
            17: 1,
            18: 0.5
        }
        
        # Find the closest predefined zoom level for the given distance
        closest_zoom = min(zoom_levels.keys(), key=lambda x: abs(zoom_levels[x] - (actual_distance )))

        #print(actual_distance,closest_zoom)
        return int(closest_zoom) -1
    

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

