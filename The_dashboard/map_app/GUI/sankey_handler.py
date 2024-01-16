import pandas as pd
import plotly.graph_objects as go

class SankeyPlotter:
    def __init__(self, dataset=None, source_column=None, target_column=None, values_column=None):
        self.dataset = dataset
        self.source_column = source_column
        self.target_column = target_column
        self.values_column = values_column

    def set_data(self, dataset, source_column, target_column, values_column):
        self.source_column = source_column
        self.target_column = target_column
        self.values_column = values_column
        self.dataset = dataset
    def get_sankey_plot(self, dataset, source_column, target_column, values_column):
        self.set_data( dataset, source_column, target_column, values_column)

        sankey_data = self.dataset.groupby([self.source_column, self.target_column])[self.values_column].sum().reset_index()

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=sankey_data[self.source_column].tolist() + sankey_data[self.target_column].tolist()
            ),
            link=dict(
                source=sankey_data[self.source_column].apply(lambda x: sankey_data[self.source_column].tolist().index(x)),
                target=sankey_data[self.target_column].apply(lambda x: len(sankey_data[self.source_column].tolist()) + sankey_data[self.target_column].tolist().index(x)),
                value=sankey_data[self.values_column]
            )
        )])

        # Update layout for better visibility
        fig.update_layout(title_text="Sankey Diagram",
                          font_size=10,
                          autosize=True,
                          margin=dict(l=20, r=20, t=20, b=20),
                          paper_bgcolor="white")

        return fig