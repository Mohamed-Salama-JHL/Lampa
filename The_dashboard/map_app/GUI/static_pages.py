import panel as pn 
from panel.theme import Native
class home_page:
    def __init__(self) -> None:
        self.design = Native
        self.page = self.create_page()
        

    def create_first_part(self):
        logo_home = pn.pane.SVG("/code/map_app/GUI/Static_data/Lampa_logo_only.svg",align=('end', 'center'), height=220,margin=(10, 50)) 
        info_text =pn.pane.Markdown('Enlighten Your Data with **LAMPA.**',styles={'font-size': '24pt','font-family': 'Microsoft Sans Serif'}) 
        info_text2 =pn.pane.Markdown('Transform your raw datasets into vibrant visual narratives with Lampa!<br /> Experience the power of data storytelling through dynamic visualizations,<br /> including Choropleth Maps, Bar Charts, Line Charts, Box Plots, and more.'
                                     ,styles={'font-size': '19pt','color':'#676767'}) 
        
        self.create_experiment_button_page = pn.widgets.Button(name='Get Started', button_type='primary', design=self.design)
        self.create_example_button_page = pn.widgets.Button(name='Demo', button_type='primary', design=self.design,button_style='outline')
        home_page_buttons_bar = pn.Row(self.create_experiment_button_page,self.create_example_button_page,align='start')
        first_part = pn.Row(pn.Column(info_text,info_text2,home_page_buttons_bar,margin=(20, 50)),logo_home, align='center',margin=(170, 10, 300, 10))
        return first_part

    def create_second_part(self):
        return pn.pane.SVG("/code/map_app/GUI/Static_data/flow_diagram.svg",align= 'center', height=360 ,margin=(100, 50)) 
    
    def create_third_part(self):
        syngenta_logo = pn.pane.SVG("/code/map_app/GUI/Static_data/Syngenta_Logo.svg",align=('end', 'center'), height=100,margin=(10, 50)) 
        jhi_logo = pn.pane.SVG("/code/map_app/GUI/Static_data/jhi_logo.svg",align=('end', 'center'), height=100,margin=(10, 50)) 
        jhl_logo = pn.pane.SVG("/code/map_app/GUI/Static_data/jhl_logo.svg",align=('end', 'center'), height=130,margin=(10, 50)) 
        logos_row = pn.Row(jhl_logo,syngenta_logo,jhi_logo,align =('center','end'),margin=(30, 10))
        logo_text =pn.pane.Markdown('**Lampa Team**',align = 'center',styles={'font-size': '22pt'}) 
        third_part = pn.Column(logo_text,pn.layout.Divider(),logos_row,align =('center','end'),margin=(180, 10))
        return third_part
    
    def create_page(self):
        first_part = self.create_first_part()
        second_part = self.create_second_part()
        third_part = self.create_third_part()

        return pn.Column(first_part,second_part,third_part                           
            ,sizing_mode='stretch_width', visible = True)

    def get_page(self):
        return self.page
    
    def get_buttons(self):
        return self.create_experiment_button_page, self.create_example_button_page
class about_page:
    def __init__(self) -> None:
        self.page = self.create_page()
    def create_page(self):
        welcome_txt = pn.pane.Markdown('Welcome to **LAMPA.**',styles={'font-size': '20pt','font-family': 'Microsoft Sans Serif'}) 
        page_text = pn.pane.HTML('''<!DOCTYPE html>
                                    <html lang="en">
                                    <head>
                                        <meta charset="UTF-8">
                                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                        <title>About Lampa</title>
                                        <style>
                                            /* Add some basic styling for positioning the SVG */
                                            svg {
                                                float: right;
                                                width: 200px; /* Adjust the width as needed */
                                                margin-left: 20px; /* Add some margin for spacing */
                                            }
                                        </style>
                                    </head>

                                    <body>

                                        <p>Lampa was developed by Mohamed Salama from James Hutton Ltd and was part of a project funded by Syngenta looking at ways in which we can develop tools to visualize and explore environmental indicator type datasets.</p>

                                        <p>Lampa had input from the following team:</p>
                                        
                                        <ul>
                                            <li>Donia Mühlematter (Syngenta)</li>
                                            <li>Mafalda Nina (Syngenta)</li>
                                            <li>Paul Shaw (Hutton)</li>
                                            <li>Sebastian Raubach (Hutton)</li>
                                            <li>Cathy Hawes (Hutton)</li>
                                        </ul>

                                        <p>and input from friends and colleagues within Information and Computational Sciences and Ecological Sciences at the James Hutton Institute in Invergowrie.</p>

                                        <p>If you have any questions or want to talk to us about Lampa then please contact Mohamed Salama on <a href="mailto:mohamed.salama@hutton.ac.uk">mohamed.salama@hutton.ac.uk</a>.</p>

                                        <p>You can also contact us by mail:</p>
                                        
                                        <address>
                                            The James Hutton Institute<br>
                                            Invergowrie<br>
                                            Dundee DD2 5DA<br>
                                            Scotland UK
                                        </address>

                                        <p>And visit our website <a href="https://www.hutton.ac.uk" target="_blank">https://www.hutton.ac.uk</a></p>

                                        <p>The source code for Lampa can be found on our GitHub page <a href="https://github.com/Mohamed-Salama-JHL/Lampa" target="_blank">https://github.com/Mohamed-Salama-JHL/Lampa</a>.<br>We promote open science and open source. If you want to be part of the project contact us and we would be glad to have you on board!</p>

                                    </body>

                                    </html>
                                        ''')
        logo = pn.pane.SVG("/code/map_app/GUI/Static_data/Lampa_logo_only.svg",align=('end', 'center'), height=220,margin=(10, 50)) 


        return pn.Column(
            welcome_txt, pn.Row(page_text, logo),
            sizing_mode='stretch_width', visible = False)
    
    def get_page(self):
        return self.page
    



class example_page:
    def __init__(self) -> None:
        self.page = self.create_page()

    def create_page(self):
        video = pn.pane.HTML('<iframe width="800" height="450" src="https://www.youtube.com/embed/eTnIPsjxOP8?si=pGvxKf7XMlEFivyC" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>',align=('center','center'),margin=(180, 10))
    

        return  pn.Column(
            video,
            sizing_mode='stretch_width', visible = False,align='center'
        )
    
    def get_page(self):
        return self.page
    