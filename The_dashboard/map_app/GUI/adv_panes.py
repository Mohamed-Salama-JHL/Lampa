import panel as pn 
from bokeh.models import Tooltip
from bokeh.models.dom import HTML


class multiselect_input:
    def __init__(self,title,tooltip_str,options=[],visible=True) -> None:
        self.title = pn.widgets.StaticText(value=title,margin = 0)
        self.tooltip_str = pn.widgets.TooltipIcon(value=Tooltip(content=HTML(self.prepare_tooltip_str(tooltip_str)), position="bottom"),margin=0)
        self.core_item = pn.widgets.MultiChoice( options=options, margin = (0, 10, 10, 10))
        self.item = pn.Column(
            pn.Row(self.title,self.tooltip_str,margin=(5, 5, 0, 10)),
            self.core_item,visible=visible
        )

    def prepare_tooltip_str(self,str):
        temp = str.split(' ')
        new_str_list = []
        for s in range(len(temp)):
            if s%5==0 and s!=0:
                new_str_list.append('<br>')
            new_str_list.append(temp[s])
        return ' '.join(new_str_list)

    def get_value(self):
        return self.core_item.value
    
    def get_item(self):
        return self.item
    
    def set_visible(self,visible=True):
        self.item.visible = visible

    def get_core_item(self):
        return self.core_item
    


class select_input:
    def __init__(self,title,tooltip_str,options=[],visible=True) -> None:
        self.title = pn.widgets.StaticText(value=title,margin = 0)
        self.tooltip_str = pn.widgets.TooltipIcon(value=Tooltip(content=HTML(self.prepare_tooltip_str(tooltip_str)), position="bottom"),margin=0)
        self.core_item = pn.widgets.Select( options=options, margin = (0, 10, 10, 10))
        self.item = pn.Column(
            pn.Row(self.title,self.tooltip_str,margin=(5, 5, 0, 10)),
            self.core_item,visible=visible
        )

    def prepare_tooltip_str(self,str):
        temp = str.split(' ')
        new_str_list = []
        for s in range(len(temp)):
            if s%5==0 and s!=0:
                new_str_list.append('<br>')
            new_str_list.append(temp[s])
        return ' '.join(new_str_list)

    def get_value(self):
        return self.core_item.value
    
    def get_item(self):
        return self.item
    
    def set_visible(self,visible=True):
        self.item.visible = visible

    def get_core_item(self):
        return self.core_item
    

class toggle_input:
    def __init__(self,title,tooltip_str,visible=True) -> None:
        self.title = pn.widgets.StaticText(value=title,margin = 0)
        self.tooltip_str = pn.widgets.TooltipIcon(value=Tooltip(content=HTML(tooltip_str), position="bottom"),margin=0)
        self.core_item = pn.widgets.Switch(name='Normalisation',margin = (0, 10, 10, 10))
        self.item = pn.Column(
            pn.Row(self.title,self.tooltip_str,margin=(5, 5, 0, 10)),
            self.core_item,visible=visible
        )

    def prepare_tooltip_str(self,str):
        pass
    def get_value(self):
        return self.core_item.value
    
    def get_item(self):
        return self.item
    
    def set_visible(self,visible=True):
        self.item.visible = visible


class number_input:
    def __init__(self,type,title,tooltip_str,start=0,end=10,step=1,value=5,visible=True) -> None:
        self.title = pn.widgets.StaticText(value=title,margin = 0)
        self.tooltip_str = pn.widgets.TooltipIcon(value=Tooltip(content=HTML(self.prepare_tooltip_str(tooltip_str)), position="bottom"),margin=0)
        self.core_item = self._create_core_item(type,start,end,step,value)
        self.item = pn.Column(
            pn.Row(self.title,self.tooltip_str,margin=(5, 5, 0, 10)),
            self.core_item,visible=visible
        )


    def prepare_tooltip_str(self,str):
        temp = str.split(' ')
        new_str_list = []
        for s in range(len(temp)):
            if s%5==0 and s!=0:
                new_str_list.append('<br>')
            new_str_list.append(temp[s])
        return ' '.join(new_str_list)
    
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