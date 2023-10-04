import pandas as pd

class data_handler:
    def __init__(self,raw_data,location_column='locations',time_column='None',value_column='value',chart_column='locations') -> None:
        self.raw_data=raw_data
        self.location_column= location_column
        self.time_column = time_column
        self.value_column = value_column
        self.chart_column=chart_column
        self.prepare_data()
        
    def prepare_data(self):
        self.data = pd.read_csv(self.raw_data,dtype={self.location_column:'string'})
        #self.data.columns= self.data.columns.str.lower()
        #self.location_column = self.location_column.lower()
        
    def pivot_data(self):
        ids_columns = [self.location_column]
        
        if 'year' in self.data.columns:
            ids_columns.append('year')
            
        pivot_data=pd.melt(self.data, id_vars=ids_columns,var_name='feature', value_name='value')
        self.data = pivot_data

    def get_data(self):
        return self.data
    
    def get_time_column(self):
        if self.time_column==None or self.time_column=='None':
            return False
        return self.time_column
    
    def get_location_column(self):
        return self.location_column
    
    def get_value_column(self):
        return self.value_column
    
    def get_chart_column(self):
        return self.chart_column