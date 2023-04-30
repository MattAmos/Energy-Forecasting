import os
import pathlib
import numpy as np
import datetime as dt
import dash
import plotly.express as px
from dash import dcc
from dash import html, callback

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from scipy.stats import rayleigh
from ui_datafinder import Forecaster

forecaster = Forecaster()

dash.register_page(__name__, name='Predictor')

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE", "drop_text": "white", 
             "drop_bg": "#082255", "drop_out": "#007ACE", "pred_colour": 'aqua', 
             "actual_colour": 'white', "base_colour": "#42C4F7"}

# DryBulb	DewPnt	WetBulb	Humidity	ElecPrice	SYSLoad	Holiday


layout = html.Div(
[
        # header
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Energy Consumption Forecaster", className="app__header__title"),
                        html.P(
                            "This home page predicts the future consumption of energy for individual households using a variety of ML models.",
                            className="app__header__title--grey",
                        ),
                    ],
                    className="app__header__desc",
                ),
            ],
            className="app__header",
        ),
        html.Div(
            [
                html.Div([
                    html.Label('Dry Bulb', style={"color": "white"}),
                    dcc.Input(id='input-box-db', type='text', value='', style={
                            'color': app_color['drop_text'],            
                            'background-color': app_color['drop_bg'], 
                            'border-color': app_color['drop_out'],    
                        },),
                ], style={'width': '24%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Dew Point', style={"color": "white"}),
                    dcc.Input(id='input-box-dp', type='text', value='', style={
                            'color': app_color['drop_text'],            
                            'background-color': app_color['drop_bg'], 
                            'border-color': app_color['drop_out'],    
                        },),
                ], style={'width': '24%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Wet Bulb', style={"color": "white"}),
                    dcc.Input(id='input-box-wb', type='text', value='', style={
                            'color': app_color['drop_text'],            
                            'background-color': app_color['drop_bg'], 
                            'border-color': app_color['drop_out'],    
                        },),
                ], style={'width': '24%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Humidity', style={"color": "white"}),
                    dcc.Input(id='input-box-hum', type='text', value='', style={
                            'color': app_color['drop_text'],            
                            'background-color': app_color['drop_bg'], 
                            'border-color': app_color['drop_out'],    
                        },),
                ], style={'width': '24%', 'display': 'inline-block'}),
            ],
        ),
        html.Div(
            [
                html.Div([
                    html.Label('Electrical Price', style={"color": "white"}),
                    dcc.Input(id='input-box-e', type='text', value='',style={
                            'color': app_color['drop_text'],            
                            'background-color': app_color['drop_bg'], 
                            'border-color': app_color['drop_out'],    
                        },),
                ], style={'width': '24%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Holiday', style={"color": "white"}),
                    dcc.Input(id='input-box-hol', type='text', value='', style={
                            'color': app_color['drop_text'],            
                            'background-color': app_color['drop_bg'], 
                            'border-color': app_color['drop_out'],    
                        },),
                ], style={'width': '24%', 'display': 'inline-block'}),

                html.Div([
                    html.Div('', id='empty_label', style={'opacity':0})
                ], style={'width': '24%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Button('Predict', id='button'),
                ], style={'width': '24%', 'display': 'inline-block'}),
            ],
        ),
        html.Div(id='output-container', children='')
    ],
    className="app__container",
)

@callback(Output('output-container', 'children'),
              Input('button', 'n_clicks'),
              State('input-box-hol', 'value'),
              State('input-box-hum', 'value'),
              State('input-box-db', 'value'),
              State('input-box-dp', 'value'),
              State('input-box-wb', 'value'),)
def update_output(n_clicks, holiday, humidity, drybulb, dewpoint, wetbulb):
    if n_clicks is not None:
        try:
            humidity = int(humidity)
            holiday = int(holiday)
            drybulb = int(drybulb)
            dewpoint = int(dewpoint)
            wetbulb = int(wetbulb)

            input = forecaster.create_input(humidity, holiday, drybulb, dewpoint, wetbulb, 10)
            prediction = forecaster.predict(input)
            print(prediction)

            # Then I need to update all the graphs i choose to put into the system after that
        except ValueError:
            print("Ensure all inputs are integer values")


