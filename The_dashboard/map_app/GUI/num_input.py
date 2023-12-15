import panel as pn 
from bokeh.models import Tooltip
from bokeh.models.dom import HTML

class number_input:
    def __init__(self,type,title,tooltip_str,start=0,end=10,step=1,value=5,visible=True) -> None:
        self.title = pn.widgets.StaticText(value=title,margin = 0)
        self.tooltip_str = pn.widgets.TooltipIcon(value=Tooltip(content=HTML(tooltip_str), position="bottom"),margin=0)
        self.core_item = self._create_core_item(type,start,end,step,value)
        self.item = pn.Column(
            pn.Row(self.title,self.tooltip_str,margin=(5, 5, 0, 10)),
            self.core_item,visible=visible
        )

    def _create_core_item(self,type,start,end,step,value):
        if type == 'int':
            return pn.widgets.IntInput(start=start, end=end, step=step, value=value)
        elif type == 'float':
            return pn.widgets.FloatInput(start=start, end=end, step=step, value=value)

    def get_value(self):
        return self.core_item.value
    
    def get_item(self):
        return self.item
    
    def set_visible(self,visible=True):
        self.item.visible = visible