import pandas as pd
import plotly.graph_objects as go


class SankeyPlotter:
    def __init__(self, dataset=None, source_column=None, target_column=None, values_column=None):
        self.dataset = dataset
        self.source_column = source_column
        self.target_column = target_column
        self.values_column = values_column
        self.selected_node = None  # Track currently selected node
    def set_data(self, dataset, source_column, target_column, values_column):
        self.source_column = source_column
        self.target_column = target_column
        self.values_column = values_column
        self.dataset = dataset
    
    def preprocess_sankey_data(self,sankey_data):
        unique_source_target = list(pd.unique(sankey_data[[self.source_column, self.target_column]].values.ravel('K')))
        mapping_dict = {k: v for v, k in enumerate(unique_source_target)}
        sankey_data[self.source_column] = sankey_data[self.source_column].map(mapping_dict)
        sankey_data[self.target_column] = sankey_data[self.target_column].map(mapping_dict)
        return unique_source_target,sankey_data
    
    def _update_link_opacity(self, link_ids, opacity):
        for link_id in link_ids:
            go.Figure.select_id("link-" + str(link_id), update={"opacity": opacity})

    def get_sankey_plot(self, dataset, source_column, target_column, values_column):
        self.set_data( dataset, source_column, target_column, values_column)

        sankey_data = self.dataset.groupby([self.source_column, self.target_column])[self.values_column].sum().reset_index()

        unique_source_target,sankey_df = self.preprocess_sankey_data(sankey_data)
        color_palette = ['#0072B2','#E69F00', '#56B4E9', '#009E73', '#F0E442' , '#CC79A7','#D55E00'
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
                        '#882255', '#6699cc', '#4477aa']*200
        fig = go.Figure(data=[go.Sankey(valueformat='r',
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=unique_source_target,
                color=color_palette
            ),
            link=dict(
                source=sankey_df[self.source_column],
                target=sankey_df[self.target_column],
                value=sankey_df[self.values_column]
            )
        )])

        fig.update_layout(
                          font_size=10,
                          autosize=True,
                          margin=dict(l=20, r=20, t=20, b=20),
                          paper_bgcolor="white")

        return fig