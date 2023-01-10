from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from app import app
from app_data import *

properties_dict = {
    'Поверхностное натяжение':interfactial_tension_organic_liquid,
    'Теплопроводность':thermal_conductivity_organic_liquid,
    'Коэффициенты объемного теплового расширения':thermal_expansion_organic_liquid,
    'Теплота парообразования':heat_vaporization_organic_liquid,
    'Давление насыщенного пара':vapor_pressure_organic_liquid,
    'Удельная теплоемкость':heat_capacity_organic_liquid,
    'Динамическая вязкость':vicosity_organic_liquid,
    'Плотность':density_organic_liquid,
    'Вязкость паров':vicosity_organic_vapor
    }

#выпадающие списки, слайдеры
properties_plot_dropdown = dcc.Dropdown(
    id='properties-plot-dropdown',
    options=[{'label':name, 'value':name} for name in density_organic_liquid.name.values],
    value=list(density_organic_liquid.name.values),
    multi=True)

property_dropdown = dcc.Dropdown(
    id='property-dropdown',
    options=[{'label':value, 'value':value} for value in list(properties_dict.keys())],
    value=list(properties_dict.keys())[0])

table_tab_layout = html.Div([
    dbc.Row(dbc.Col([dbc.Table.from_dataframe(df=ph_organic, index=True)])),
    html.Hr(),
    dbc.Row([dbc.Col([html.Div(properties_plot_dropdown)], width={"size": 8, "offset": 0}),
             dbc.Col([html.Div(property_dropdown)], width={"size": 4, "offset": 0})]),
    dbc.Row([dbc.Col([dcc.Graph(id='property-figure')], width={"size": 10, "offset": 1})]),
    
    ])

@app.callback(
    Output('properties-plot-dropdown', 'value'),
    Output('properties-plot-dropdown', 'options'),
    Input('property-dropdown', 'value'),
)
def pull_substances_list(property_value):
    global properties_dict    
    return (list(properties_dict[property_value].name.values),
            [{'label':name, 'value':name} for name in properties_dict[property_value].name.values])

@app.callback(
    Output('property-figure', 'figure'),
    Input('property-dropdown', 'value'),
    Input('properties-plot-dropdown', 'value'),
)
def create_propery_plot(property_value, substances):
    
    global properties_dict
    
    property = properties_dict[property_value]
    
    def get_organic_liquid_frame(substances, property):
        property.index = property.name
        property = property.drop('name', axis=1)
        return property.loc[substances]
    
    def property_title_values(property):
        if property is interfactial_tension_organic_liquid:
            return 'мДж/м^2'
        elif property is thermal_conductivity_organic_liquid:
            return 'Вт/(м*K)'
        elif property is thermal_expansion_organic_liquid:
            return 'b*10^3, K^-1'
        elif property is heat_vaporization_organic_liquid:
            return 'кДж/кг'
        elif property is vapor_pressure_organic_liquid:
            return 'мм.рт.ст.'
        elif property is heat_capacity_organic_liquid:
            return  'Дж/(кг*K)'
        elif property is vicosity_organic_liquid:
            return 'мПа*с'
        elif property is density_organic_liquid:
            return 'кг/м^3'
        elif property is vicosity_organic_vapor:
            return 'мкПа*с'
        else:
            return 'Ошибка'

    df = get_organic_liquid_frame(substances, property)

    fig = go.Figure()

    #add traces
    for substance in substances:
        fig.add_trace(go.Scatter(x=df.loc[substance].keys(), y=df.loc[substance].values, name=substance), )

    fig.update_xaxes(title_text='Температура, °С',
                                gridcolor='rgb(105,105,105)',
                                griddash='1px',
                                zeroline=False)
    fig.update_yaxes(title_text=property_title_values(property),
                    gridcolor='rgb(105,105,105)',
                    griddash='1px',
                    zeroline=False)

    #настраиваем график снаружи и на границах
    fig.update_xaxes(                
                    showline=True, linewidth=2, linecolor='black',
                    mirror=True,
                    ticks='inside')
    fig.update_yaxes(
                    showline=True, linewidth=2, linecolor='black',
                    mirror=True,
                    ticks='inside')
    fig.update_layout(
                    autosize=True,
                    margin=dict(l=20, r=5, t=20, b=2),
                    showlegend=False,
                    plot_bgcolor='white')
    
    return fig