from panel.layout.gridstack import GridStack
import warnings
import panel as pn
import plotly.express as px
import plotly.graph_objs as go
import time
pn.extension('gridstack')
warnings.filterwarnings("error")


class grid_stack:
    def __init__(self,charts_list=[],ncols=6,nrows=22,min_height=3000,min_width=1500,name='Dashboard') -> None:
        self.ncols = ncols
        self.nrows = nrows
        self.min_height = min_height
        self.min_width = min_width
        self.gstack = GridStack(sizing_mode='stretch_both', min_height=min_height,mode ='error',ncols=self.ncols,nrows=self.nrows, name = name,visible = True,min_width=min_width)
        self.external_gstack = pn.Column(self.gstack, name = name,visible = False)
        self.add_charts(charts_list)
    
    def add_charts(self,charts_list=[],x_size=3,y_size=2):
        for chart_obj in charts_list:
            self.add_chart(chart_obj,x_size=x_size,y_size=y_size)
            
    def add_chart(self,chart_obj,x_size=3,y_size=2,reverse=False):
        place = self.find_place(self.gstack.grid,x_size,y_size,reverse)
        if place:
            def update_grid():
                self.gstack[place[1]:place[1]+y_size, place[0]:place[0]+x_size] = chart_obj
            pn.state.execute(update_grid)
            return True
        return False
        
    def find_place(self,grid,x_size=3,y_size=2,reverse=False):
        rows = len(grid)
        cols = len(grid[0])

        def is_empty_block(x, y):
            for i in range(x, x + x_size):
                for j in range(y, y + y_size):
                    if i >= cols or j >= rows or grid[j][i] != 0:
                        return False
            return True

        best_location = None
        if not reverse:
            for i in range(cols - x_size + 1):
                for j in range(rows - y_size + 1):
                    if is_empty_block(i, j):
                        best_location = (i, j)
                        return best_location  
        elif  reverse:
            for j in range(rows - y_size + 1):
                for i in range(cols - x_size + 1):
                    if is_empty_block(i, j):
                        best_location = (i, j)
                        return best_location  

        return best_location
    def remove_chart(self,name):
        for i in range(self.nrows):
            for j in range(self.ncols):
                try:
                    if self.gstack[i,j].name == name:
                        del self.gstack[i,j]
                        break
                except:
                    pass
    def get_gridstack(self):
        return self.external_gstack
    
    def refresh_grid_stack(self):
        '''
        temp = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='temp', margin=2),scroll=False)
        self.add_chart(temp,1,1)
        time.sleep(1)
        self.remove_chart(temp.name)

        '''

        temp = self.gstack
        self.external_gstack.clear()
        self.external_gstack.append(temp)
        self.gstack = temp 

        