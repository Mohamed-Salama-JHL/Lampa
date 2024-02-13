import numpy as np
import param
import panel as pn
import numpy as np
import hvplot.dask
import pandas as pd
import panel as pn
import hvplot.pandas
import pandas as pd
import numpy as np
from io import StringIO
from urllib.request import urlopen
import json
from panel.theme import Native
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score

import plotly.express as px
import plotly.graph_objs as go
import logging
from sklearn.preprocessing import MinMaxScaler
from .num_input import *
from .gridstack_handler import grid_stack
from .analysis_page_abstract import analysis_abstract
from .data_handler import data_handler
from .adv_panes import *

pd.options.mode.chained_assignment = None 
logging.basicConfig( level=logging.ERROR,force=True)

pn.extension('floatpanel')
pn.extension('gridstack')
pn.extension('tabulator')
pn.extension('plotly')
pn.extension(notifications=True)
pn.extension(design='bootstrap', template='material' )

class classification_module:
    
    def __init__(self,dataset) -> None:
        self.dataset = dataset#.reset_index()
        self.working_dataset = None
        self.output_dataset = None
        self.reduction_dataset = None
        self.algorithms_list = ['Logistic Regression','Decision Tree', 'Random Forest']
        self.classification_page = None
        self.actual_fig_flag = False
        self.main_page_created = False
        self.scaler = None
        self.model = None
        self.testing_dataset =None
    
    def get_updated_dataset_button(self):
        return self.update_results_button

    def set_dataset(self,new_dataset):
        self.dataset = new_dataset
    
    def get_page(self):
        if self.classification_page == None:
            self.create_page()
        return self.classification_page
    
    def get_controls(self):
        if self.classification_page == None:
            self.create_page()
        return self.classification_controls
    
    def get_main_area(self):
        if self.classification_page == None:
            self.create_page()
        
        return self.classification_main_area
    
    def create_page(self):
        self.create_sidebar()
        self.create_main_area()
        self.bend_components_actions()
        self.classification_page = pn.Column(
            pn.Row(
                self.classification_main_area,
                self.classification_controls,
                #sizing_mode='stretch_width'
            ),
            sizing_mode='stretch_both',name='classification')
    
    def create_sidebar (self):
        self.create_algorithm_settings_component()
        self.create_dataset_setting_component()
        self.create_control_buttons()
        self.create_general_widgets()
        self.classification_controls = pn.Card(pn.Column(self.algo_settings_card,self.data_settings_card,self.controls_buttons_row,self.loading), title="<h1 style='font-size: 15px;'>Classification Settings</h1>", styles={"border": "none", "box-shadow": "none"})
    
    def get_grid_stack(self,chart_list):
        self.grid_stack_handler = grid_stack(chart_list,min_width=950,min_height=2000,ncols=6,nrows = 15, name='Classification')
        return self.grid_stack_handler.get_gridstack()
    

    #Need to change
    def create_main_area(self):
        
        self.feature_importance = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='Feature importance', margin=2,width=600),scroll=False,visible = False)
        self.training_data = pn.Column(pn.pane.Plotly(go.Figure().update_layout(template="plotly_white"),name='Training data', margin=2,width=600),scroll=False,visible = False)
        self.Accuracy = pn.indicators.Number(name='Accuracy', value=0, format='{value}', font_size= '20pt')
        self.Recall = pn.indicators.Number(name='Recall', value=0, format='{value}', font_size= '20pt')
        self.Precision = pn.indicators.Number(name='Precision', value=0, format='{value}', font_size= '20pt')
        self.matrics_values = pn.Row(self.Accuracy,self.Recall,self.Precision)
        chart_list= [self.feature_importance,self.matrics_values] 
        self.classification_main_area = self.get_grid_stack(chart_list)
        self.classification_main_area.visible = True

    #Need to change
    def create_algorithm_settings_component(self):
        self.select_algorithm = pn.widgets.Select(name='Classification Algorithms', options=self.algorithms_list, value ='Logistic Regression' )
        self.select_max_depth = number_input(type='int',title='Max Depth: ',tooltip_str='Max depth of the decision tree', start=1, end=10000, step=1, value=3,visible=False)
        self.select_max_iteration = number_input(type='int',title='Max Iteration: ',tooltip_str='Max iteration for logistic regression', start=100, end=100000, step=1, value=100,visible=True)
        self.select_test_perc = number_input(type='int',title='Testing Data Percentage: ',tooltip_str='The percentage of the dataset designated for testing the model.', start=1, end=100, step=1, value=20,visible=True)

        self.algo_settings_card = pn.Column(self.select_algorithm, self.select_test_perc.get_item(),self.select_max_iteration.get_item(),self.select_max_depth.get_item())#, title="<h1 style='font-size: 15px;'>Algorithm settings</h1>", styles={"border": "none", "box-shadow": "none"}

    
    def create_dataset_setting_component(self):
        numric_columns = list(self.dataset.select_dtypes(include=['number']).columns)
        self.select_classification_columns = pn.widgets.MultiChoice(name= 'Classification Columns' ,options=numric_columns)
        self.select_classification_target = pn.widgets.Select(name= 'Classification Target' ,options=numric_columns)
        self.check_normalization = toggle_input('Normalisation: ','Min-Max normalisation for the features',visible=True)
        
        self.data_settings_card = pn.Column(self.select_classification_target,self.select_classification_columns,self.check_normalization.get_item())#, title="<h1 style='font-size: 15px;'>Dataset settings</h1>", styles={"border": "none", "box-shadow": "none"})

    def create_control_buttons(self):
        self.run_classification_button = pn.widgets.Button(name='Run Classification', button_type='primary',margin = (5, 3, 5, 15))
        self.update_results_button =  pn.widgets.Button(name='Update Results', button_type='primary',margin = (5, 3, 5, 3))
        self.download_classification_data_button = pn.widgets.FileDownload(callback=pn.bind(self.get_classification_data_io), filename='classification_data.csv', label = 'Download Dataset',align = 'center',button_style='outline',button_type='primary',height=40 )
        self.freeze_dashboard = pn.widgets.Toggle(button_type='primary', button_style='outline', icon='snowflake', align='center', icon_size='14px',margin = (5, 3, 5, 3))     
        self.test_data_input = pn.widgets.FileInput(name= 'Upload Testing', accept='.csv,.xlsx',design=Native,margin = (10, 25, 10, 25))
        self.controls_buttons_row = pn.Column(pn.Row(self.run_classification_button,self.update_results_button,self.freeze_dashboard,sizing_mode='stretch_width'),self.test_data_input,self.download_classification_data_button,sizing_mode='stretch_width')
    
    def create_general_widgets(self):
        self.loading = pn.indicators.LoadingSpinner(value=True, size=20, name='Loading...', visible=False)
    
    def run_classification(self,event=None):
        self.start_classification()    
        self.dataset_preprocessing()
        if self.select_algorithm.value=='Logistic Regression':
            feature_importance,mae,mse,rmse = self.run_lr()
        elif self.select_algorithm.value=='Decision Tree':
            feature_importance,mae,mse,rmse=self.run_dt()
        elif self.select_algorithm.value=='Random Forest':
            feature_importance,mae,mse,rmse=self.run_rf()
        self.create_charts_objects(feature_importance,mae,mse,rmse)
        self.end_classification()
    
    def start_classification(self):
        self.loading.visible= True
        self.run_classification_button.disabled = True
    def end_classification(self):
        self.loading.visible= False
        self.run_classification_button.disabled = False
    
    def dataset_preprocessing(self):
        self.classification_features_list = self.select_classification_columns.value if len(self.select_classification_columns.value)>=1 else self.select_classification_columns.options
        self.classification_target_feature = self.select_classification_target.value
        self.classification_features_list = [feature for feature in self.classification_features_list if feature!=self.classification_target_feature]
        self.classification_all_features = [feature for feature in self.classification_features_list]
        self.classification_all_features.append(self.classification_target_feature)
        self.output_dataset = self.dataset[self.classification_all_features]
        self.missing_values_handling()
        self.target_traing_series = self.output_dataset[self.classification_target_feature]
        self.output_dataset = self.output_dataset[self.classification_features_list]
        non_numeric_columns = self.output_dataset.select_dtypes(exclude=['number']).columns
        self.categorical_flag = False
        
        if not non_numeric_columns.empty:
            self.working_dataset = pd.get_dummies(self.output_dataset, columns=non_numeric_columns)
            self.categorical_flag = True
        else:
            self.working_dataset = self.output_dataset

        if self.check_normalization.get_value():
            self.normalize_dataset()

        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.working_dataset, self.target_traing_series, test_size=(self.select_test_perc.get_value()/100), random_state=42)


    #show table with nan values of the target if exist
    #handle missing from feature the same as dataset and the opposite
    
    def normalize_dataset(self):
        self.scaler = MinMaxScaler()
        self.working_dataset = self.scaler.fit_transform(self.working_dataset)


    def missing_values_handling(self):
        self.output_dataset.dropna(inplace=True)

    def run_lr(self):
        self.model = LogisticRegression(multi_class='auto',max_iter=self.select_max_iteration.get_value())
        try:
            self.model.fit(self.X_train, self.y_train)
            y_pred = self.model.predict(self.X_test)
            feature_importance = self.model.coef_
            accuracy = accuracy_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred, average='macro',zero_division = np.nan)
            precision = precision_score(self.y_test, y_pred, average='macro',zero_division = np.nan)
            self.output_dataset.loc[:, 'classification']=self.model.predict(self.working_dataset)
            avg_importance = np.mean(np.abs(feature_importance), axis=0)

        except:
            empty_imp = [0 for i in self.output_dataset.columns if i != 'classification']
            pn.state.notifications.error(f'<span style="font-family: sans-serif; font-size: 15px;">Logistic regression failed to converge, try different parameters.</span>',duration=0)
            return empty_imp,0,0,0
        return avg_importance,accuracy,precision,recall

    def run_dt(self):
        self.model = DecisionTreeClassifier(max_depth=self.select_max_depth.get_value())
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        feature_importance = self.model.feature_importances_
        accuracy = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred, average='macro',zero_division = np.nan)
        precision = precision_score(self.y_test, y_pred, average='macro',zero_division = np.nan)
        self.output_dataset.loc[:, 'classification']=self.model.predict(self.working_dataset)
        return feature_importance,accuracy,precision,recall
    
    def run_rf(self):
        self.model = RandomForestClassifier(max_depth=self.select_max_depth.get_value())
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        feature_importance = self.model.feature_importances_
        accuracy = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred, average='macro',zero_division = np.nan)
        precision = precision_score(self.y_test, y_pred, average='macro',zero_division = np.nan)
        self.output_dataset.loc[:, 'classification']=self.model.predict(self.working_dataset)
        return feature_importance,accuracy,precision,recall

    def create_charts_objects(self,feature_importance,accuracy,precision,recall):
        feature_importance_fig = None
        self.feature_importance.visible = True
        #self.reduction_dataset["classification"] = self.reduction_dataset["classification"].astype(str)

        feature_names = [i for i in self.output_dataset.columns if i != 'classification']

        
        coeff_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': feature_importance})
        feature_importance_fig = px.bar(coeff_df, x='Feature', y='Coefficient', title='Linear classification Coefficients',template="plotly_white").update_layout(margin=dict(l=20, r=20, t=50, b=5),)
        
        self.Accuracy.value = round(accuracy, 3)
        self.Recall.value = round(recall, 3)
        self.Precision.value = round(precision, 3)

        feature_importance_fig.layout.autosize = True
        self.update_charts(feature_importance_fig)
            
    def update_charts(self,output_fig):
        self.feature_importance.clear()
        self.feature_importance.append(pn.pane.Plotly(output_fig,name='Feature importance',  margin=2)) 

    def show_lr_settings(self):
        self.select_max_depth.set_visible(False)
        self.select_max_iteration.set_visible(True)
                 
    def show_dt_settings(self):
        self.select_max_depth.set_visible(True)
        self.select_max_iteration.set_visible(False)
    
    def show_rf_settings(self):
        self.select_max_depth.set_visible(True)
        self.select_max_iteration.set_visible(False)
    
    def algo_settings_show(self,event):
        if self.select_algorithm.value == 'Logistic Regression':
            self.show_lr_settings()
        elif self.select_algorithm.value == 'Decision Tree':
            self.show_dt_settings()
        elif self.select_algorithm.value == 'Random Forest':
            self.show_rf_settings()

    
    def freezing_dashboard(self,event):
        dynamic_flag = not self.freeze_dashboard.value
        self.grid_stack_handler.dynamic(dynamic_flag)

    def bend_components_actions(self):
        self.run_classification_button.param.watch(self.run_classification,'value')
        self.select_algorithm.param.watch(self.algo_settings_show,'value')
        self.freeze_dashboard.param.watch(self.freezing_dashboard,'value')
        self.test_data_input.param.watch(self.test_data_handler,'value')

    def get_classification_data_io(self):
        sio = StringIO()
        comment = f'This results are done by {self.select_algorithm.value}'
        self.process_testing_dataset()
        try:
            if comment:
                sio.write("# " + comment + "\n")
            
            if self.process_testing_dataset():
                temp = self.testing_output_dataset.copy()
                temp.to_csv(sio)
                sio.seek(0)
            else:
                temp = self.output_dataset.copy()
                temp[self.classification_target_feature] = self.target_traing_series
                temp.to_csv(sio)
                sio.seek(0)
        
        except:
            pass
        return sio

    def get_trigger_button(self):
        return self.update_results_button
    
    def get_classification_column(self):
        try:
            return self.output_dataset['classification']
        except:
            return None
        
    def refresh_main_page(self):
        self.grid_stack_handler.refresh_grid_stack()


    def test_data_handler(self,event):
        self.data_handler = data_handler(raw_data=self.test_data_input.value,file_name=self.test_data_input.filename)
        temp_testing_dataset = self.data_handler.get_data()
        if self.validate_testing_dataset(temp_testing_dataset):
            self.testing_dataset = temp_testing_dataset

    def validate_testing_dataset(self,dataset):
        if isinstance(dataset, type(None)):
            return False
        for column in dataset.columns:
            if column not in self.select_classification_columns.value or dataset[column].dtypes != self.dataset[column].dtypes:
                return False
        return True
            
    def process_testing_dataset(self):

        if not isinstance(self.testing_dataset, pd.DataFrame) or self.model ==None or not self.validate_testing_dataset(self.testing_dataset):
            print('not working',isinstance(self.testing_dataset, type(None)),self.model ==None, self.validate_testing_dataset(self.testing_dataset))
            return False
        self.testing_output_dataset = self.testing_dataset[self.classification_all_features]
        self.testing_output_dataset.dropna(inplace=True)
        non_numeric_columns = self.testing_output_dataset.select_dtypes(exclude=['number']).columns
        self.categorical_flag = False
        
        if not non_numeric_columns.empty:
            self.working_dataset = pd.get_dummies(self.testing_output_dataset, columns=non_numeric_columns)
            self.categorical_flag = True
        else:
            self.testing_working_dataset = self.testing_output_dataset

        if self.check_normalization.get_value():
            self.testing_working_dataset = self.scaler.transform(self.testing_working_dataset)

        self.testing_output_dataset.loc[:, 'classification']=self.model.predict(self.testing_working_dataset)
        return True