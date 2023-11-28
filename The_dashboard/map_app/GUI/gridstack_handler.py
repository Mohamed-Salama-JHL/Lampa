from panel.layout.gridstack import GridStack
import warnings
import panel as pn

pn.extension('gridstack')
warnings.filterwarnings("error")


class grid_stack:
    def __init__(self,charts_list=[],ncols=6,nrows=22,min_height=3000,min_width=1500) -> None:
        self.ncols = ncols
        self.nrows = nrows
        self.min_height = min_height
        self.min_width = min_width
        self.gstack = GridStack(sizing_mode='stretch_both', min_height=3000,mode ='error',ncols=self.ncols,nrows=self.nrows, name = 'Dashboard',visible = False,min_width=1500)
        self.add_charts(charts_list)
    
    def add_charts(self,charts_list=[]):
        for chart_obj in charts_list:
            self.add_chart(chart_obj)
            
    def add_chart(self,chart_obj,x_size=3,y_size=2):
        place = self.find_place(self.gstack.grid,x_size,y_size)
        if place:
            def update_grid():
                self.gstack[place[1]:place[1]+y_size, place[0]:place[0]+x_size] = chart_obj
            pn.state.execute(update_grid)
            return True
        return False
        
    def find_place(self,grid,x_size=3,y_size=2):
        rows = len(grid)
        cols = len(grid[0])

        def is_empty_block(x, y):
            for i in range(x, x + x_size):
                for j in range(y, y + y_size):
                    if i >= cols or j >= rows or grid[j][i] != 0:
                        return False
            return True

        best_location = None

        for i in range(cols - x_size + 1):
            for j in range(rows - y_size + 1):
                if is_empty_block(i, j):
                    best_location = (i, j)
                    return best_location  # Return the first valid block found

        return best_location
    def remove_chart(self,name):
        for i in range(self.nrows):
            for j in range(self.ncols):
                try:
                    #print(type(self.gstack[i,j]),(i,j),self.gstack[i,j].name)
                    if self.gstack[i,j].name == name:
                        #print('here at',i,j)
                        del self.gstack[i,j]
                        break
                except:
                    pass
    def get_gridstack(self):
        return self.gstack