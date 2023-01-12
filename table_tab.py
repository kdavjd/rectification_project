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

vicosity_dropdown = dcc.Dropdown(
    id='vicosity-dropdown',
    options=[{'label':name, 'value':name} for name in list(vicosity_solution['salt'].values)],
    value=list(vicosity_solution['salt'].values),
    multi=True)
vicosity_3d_dropdown = dcc.Dropdown(
    id='vicosity-3d-dropdown',
    options=[{'label':name, 'value':name} for name in list(vicosity_solution['salt'].values)],
    value=list(vicosity_solution['salt'].values),
    )
density_dropdown = dcc.Dropdown(
    id='density-dropdown',
    options=[{'label':name, 'value':name} for name in list(density_solution['salt'].values)],
    value=list(density_solution['salt'].values),
    multi=True)
density_3d_dropdown = dcc.Dropdown(
    id='density-3d-dropdown',
    options=[{'label':name, 'value':name} for name in list(density_solution['salt'].values)],
    value=list(density_solution['salt'].values),
    )
specific_heat_capacity_dropdown = dcc.Dropdown(
    id='specific-heat-capacity-dropdown',
    options=[{'label':name, 'value':name} for name in list(specific_heat_capacity_solution['salt'].values)],
    value=list(specific_heat_capacity_solution['salt'].values),
    multi=True)
specific_heat_capacity_3d_dropdown = dcc.Dropdown(
    id='specific-heat-capacity-3d-dropdown',
    options=[{'label':name, 'value':name} for name in list(specific_heat_capacity_solution['salt'].values)],
    value=list(specific_heat_capacity_solution['salt'].values),
    )
thermal_conductivity_dropdown = dcc.Dropdown(
    id='thermal-conductivity-dropdown',
    options=[{'label':name, 'value':name} for name in list(thermal_conductivity_solutions['salt'].values)],
    value=list(thermal_conductivity_solutions['salt'].values),
    multi=True)
thermal_conductivity_3d_dropdown = dcc.Dropdown(
    id='thermal-conductivity-3d-dropdown',
    options=[{'label':name, 'value':name} for name in list(thermal_conductivity_solutions['salt'].values)],
    value=list(thermal_conductivity_solutions['salt'].values),
    )
vicosity_solution_w_slider = html.Div([
    dcc.RangeSlider(0, 2, 0.1, value=[0.3], marks={
        0: {'label': '0'},        
        0.5: {'label': '0.5'},
        1: {'label': '1'},
        1.5: {'label': '1.5'},
        2: {'label': '2'},
    },
                    id='vicosity-w-slider')
])
vicosity_solution_t_slider = html.Div([
    dcc.RangeSlider(0, 150, 1, value=[0,100], marks={        
        0: {'label': '0°C'},
        25: {'label': '25°C'},
        50: {'label': '50°C'},
        75: {'label': '75°C'},
        100: {'label': '100°C'},
        125: {'label': '125°C'},
        150: {'label': '150°C'},
    },
                    id='vicosity-t-slider')
])
density_solution_w_slider = html.Div([
    dcc.RangeSlider(0, 2, 0.1, value=[0.3], marks={
        0: {'label': '0'},        
        0.5: {'label': '0.5'},
        1: {'label': '1'},
        1.5: {'label': '1.5'},
        2: {'label': '2'},
    },
                    id='density-w-slider')
])
density_solution_t_slider = html.Div([
    dcc.RangeSlider(0, 150, 1, value=[0,100], marks={        
        0: {'label': '0°C'},
        25: {'label': '25°C'},
        50: {'label': '50°C'},
        75: {'label': '75°C'},
        100: {'label': '100°C'},
        125: {'label': '125°C'},
        150: {'label': '150°C'},
    },
                    id='density-t-slider')
])
specific_heat_capacity_solution_w_slider = html.Div([
    dcc.RangeSlider(0, 2, 0.1, value=[0.3], marks={
        0: {'label': '0'},        
        0.5: {'label': '0.5'},
        1: {'label': '1'},
        1.5: {'label': '1.5'},
        2: {'label': '2'},
    },
                    id='specific-heat-capacity-w-slider')
])
specific_heat_capacity_solution_t_slider = html.Div([
    dcc.RangeSlider(0, 150, 1, value=[0,100], marks={        
        0: {'label': '0°C'},
        25: {'label': '25°C'},
        50: {'label': '50°C'},
        75: {'label': '75°C'},
        100: {'label': '100°C'},
        125: {'label': '125°C'},
        150: {'label': '150°C'},
    },
                    id='specific-heat-capacity-t-slider')
])
thermal_conductivity_solution_w_slider = html.Div([
    dcc.RangeSlider(0, 2, 0.1, value=[0.3], marks={
        0: {'label': '0'},        
        0.5: {'label': '0.5'},
        1: {'label': '1'},
        1.5: {'label': '1.5'},
        2: {'label': '2'},
    },
                    id='thermal-conductivity-w-slider')
])
thermal_conductivity_solution_t_slider = html.Div([
    dcc.RangeSlider(0, 150, 1, value=[0,100], marks={        
        0: {'label': '0°C'},
        25: {'label': '25°C'},
        50: {'label': '50°C'},
        75: {'label': '75°C'},
        100: {'label': '100°C'},
        125: {'label': '125°C'},
        150: {'label': '150°C'},
    },
                    id='thermal-conductivity-t-slider')
])


table_tab_layout = html.Div([
    dbc.Row(dbc.Col([dbc.Table.from_dataframe(df=ph_organic, index=True)])),
    html.Hr(),
    dbc.Row([dbc.Col([html.H4('Основные физико-химические свойства некоторых органических веществ от температуры'),])]),
    dbc.Row([dbc.Col([html.Div(properties_plot_dropdown)], width={"size": 8, "offset": 0}),
             dbc.Col([html.Div(property_dropdown)], width={"size": 4, "offset": 0})]),
    dbc.Row([dbc.Col([dcc.Graph(id='property-figure')], width={"size": 10, "offset": 1})]),
    html.Hr(),
    dbc.Row([dbc.Col([html.H5('Вязкозть водных растворов некоторых неорганических веществ от температуры и массовой доли'),
                      html.Div(vicosity_dropdown)], width={"size": 10, "offset": 0}),
             dbc.Col([html.Div('Выберите водный раствор для 3d графика'),
                      html.Div(vicosity_3d_dropdown),], width={"size": 2, "offset": 0})]),
    dbc.Row([dbc.Col([html.Div('Массовая доля, кг/кг'),
                      html.Div(vicosity_solution_w_slider)], width={"size": 3, "offset": 0}),
             dbc.Col([html.Div('Диапазон температуры, °С'),
                      html.Div(vicosity_solution_t_slider)], width={"size": 3, "offset": 0})]),
    dbc.Row([dbc.Col([dcc.Graph(id='vicosity-solution-figure')], width={"size": 6, "offset": 0}),
             dbc.Col([dcc.Graph(id='vicosity-solution-3d-figure')], width={"size": 5, "offset": 1})]),
    html.Hr(),
    dbc.Row([dbc.Col([html.H5('Плотность водных растворов некоторых неорганических веществ от температуры и массовой доли'),
                      html.Div(density_dropdown)], width={"size": 10, "offset": 0}),
             dbc.Col([html.Div('Выберите водный раствор для 3d графика'),
                      html.Div(density_3d_dropdown)], width={"size": 2, "offset": 0})]),
    dbc.Row([dbc.Col([html.Div('Массовая доля, кг/кг'),
                      html.Div(density_solution_w_slider)], width={"size": 3, "offset": 0}),
             dbc.Col([html.Div('Диапазон температуры, °С'),
                      html.Div(density_solution_t_slider)], width={"size": 3, "offset": 0})]),
    dbc.Row([dbc.Col([dcc.Graph(id='density-solution-figure')], width={"size": 6, "offset": 0}),
             dbc.Col([dcc.Graph(id='density-solution-3d-figure')], width={"size": 5, "offset": 1})]),
    html.Hr(),
    dbc.Row([dbc.Col([html.H5('Удельная теплоёмкость водных растворов некоторых неорганических веществ от температуры и массовой доли'),
                      html.Div(specific_heat_capacity_dropdown)], width={"size": 10, "offset": 0}),
             dbc.Col([html.Div('Выберите водный раствор для 3d графика'),
                      html.Div(specific_heat_capacity_3d_dropdown)], width={"size": 2, "offset": 0})]),
    dbc.Row([dbc.Col([html.Div('Массовая доля, кг/кг'),
                      html.Div(specific_heat_capacity_solution_w_slider)], width={"size": 3, "offset": 0}),
             dbc.Col([html.Div('Диапазон температуры, °С'),
                      html.Div(specific_heat_capacity_solution_t_slider)], width={"size": 3, "offset": 0})]),
    dbc.Row([dbc.Col([dcc.Graph(id='specific-heat-capacity-solution-figure')], width={"size": 6, "offset": 0}),
             dbc.Col([dcc.Graph(id='specific-heat-capacity-solution-3d-figure')], width={"size": 5, "offset": 1})]),
    html.Hr(),
    dbc.Row([dbc.Col([html.H5('Теплопроводность водных растворов некоторых неорганических веществ от температуры и массовой доли'),
                      html.Div(thermal_conductivity_dropdown)], width={"size": 10, "offset": 0}),
             dbc.Col([html.Div('Выберите водный раствор для 3d графика'),
                      html.Div(thermal_conductivity_3d_dropdown)], width={"size": 2, "offset": 0})]),
    dbc.Row([dbc.Col([html.Div('Массовая доля, кг/кг'),
                      html.Div(thermal_conductivity_solution_w_slider)], width={"size": 3, "offset": 0}),
             dbc.Col([html.Div('Диапазон температуры, °С'),
                      html.Div(thermal_conductivity_solution_t_slider)], width={"size": 3, "offset": 0})]),
    dbc.Row([dbc.Col([dcc.Graph(id='thermal-conductivity-solution-figure')], width={"size": 6, "offset": 0}),
             dbc.Col([dcc.Graph(id='thermal-conductivity-solution-3d-figure')], width={"size": 5, "offset": 1})]),
    html.Hr(),
    
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

@app.callback(
    Output('vicosity-solution-figure', 'figure'),
    Input('vicosity-dropdown', 'value'),    
    Input('vicosity-t-slider', 'value'),
    Input('vicosity-w-slider', 'value'),
)
def plot_vicosity_figure(substances_list: list[str], t: list[str], w: str):
    
    def get_plot_values(name: str, _t: list[int], w: float):
        
        def get_vicosity_solution(name, w, t, uaq):
            component = vicosity_solution[vicosity_solution['salt'] == name].fillna(0)
            return uaq*np.exp(w*(component.b1.values + component.b2.values*1e-2 * t - component.b3.values*1e-7 * t**2))
        
        t = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= _t[0]) & (aqua_liquid_saturation['temperature'] <= _t[1])]['temperature']
        uaq = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= _t[0]) & (aqua_liquid_saturation['temperature'] <= _t[1])]['viscosity_kilo'].values/1000    
        u = [float(*get_vicosity_solution(name, w, t_, uaq_)) for t_, uaq_ in list(zip(t,uaq))]
        
        return pd.DataFrame({'temperature':list(t),
                             'vicosity':u})
    
    temperature = list(map(lambda x: int(x), t))
    w = float(*w)
    
    fig = go.Figure()
    for substance in substances_list:        
        df = get_plot_values(substance, temperature, w)        
        fig.add_trace(go.Scatter(x=df['temperature'], y=df['vicosity'], name=substance))
    
    fig.update_xaxes(title_text='Температура, °С',
                            gridcolor='rgb(105,105,105)',
                            griddash='1px',
                            zeroline=False)
    fig.update_yaxes(title_text='мПа*с',
                    gridcolor='rgb(105,105,105)',
                    griddash='1px',
                    zeroline=False)

    #настраиваем график снаружи и на границах
    fig.update_xaxes(range=[temperature[0], temperature[1]+1],
                    showline=True, linewidth=2, linecolor='black',
                    mirror=True,
                    ticks='inside')
    fig.update_yaxes(
                    showline=True, linewidth=2, linecolor='black',
                    mirror=True,
                    ticks='inside')
    fig.update_layout(
                    autosize=False,
                    margin=dict(l=20, r=5, t=20, b=2),
                    showlegend=False,
                    plot_bgcolor='white')
    
    return fig

@app.callback(
    Output('density-solution-figure', 'figure'),
    Input('density-dropdown', 'value'),    
    Input('density-t-slider', 'value'),
    Input('density-w-slider', 'value'),
)
def plot_density_figure(substances_list: list[str], t: list[str], w: str):
    
    def get_plot_values(name: str, _t: list[int], w: float):
        
        def get_density_solution(name, w, t, paq):
            component = density_solution[density_solution['salt'] == name].fillna(0)            
            return paq*np.exp(w*(component.a1 + component.a2*1e-4 * t - component.a3*1e-6 * t**2))
                
        t = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= _t[0]) & (aqua_liquid_saturation['temperature'] <= _t[1])]['temperature']
        paq = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= _t[0]) & (aqua_liquid_saturation['temperature'] <= _t[1])]['density'].values
        p = [float(*get_density_solution(name, w, t_, paq_)) for t_, paq_ in list(zip(t,paq))]
        
        return pd.DataFrame({'temperature':list(t),
                             'density':p})
    
    temperature = list(map(lambda x: int(x), t))
    w = float(*w)
    
    fig = go.Figure()
    for substance in substances_list:        
        df = get_plot_values(substance, temperature, w)        
        fig.add_trace(go.Scatter(x=df['temperature'], y=df['density'], name=substance))
    
    fig.update_xaxes(title_text='Температура, °С',
                            gridcolor='rgb(105,105,105)',
                            griddash='1px',
                            zeroline=False)
    fig.update_yaxes(title_text='кг/М^3',
                    gridcolor='rgb(105,105,105)',
                    griddash='1px',
                    zeroline=False)

    #настраиваем график снаружи и на границах
    fig.update_xaxes(range=[temperature[0], temperature[1]+1],
                    showline=True, linewidth=2, linecolor='black',
                    mirror=True,
                    ticks='inside')
    fig.update_yaxes(
                    showline=True, linewidth=2, linecolor='black',
                    mirror=True,
                    ticks='inside')
    fig.update_layout(
                    autosize=False,
                    margin=dict(l=20, r=5, t=20, b=2),
                    showlegend=False,
                    plot_bgcolor='white')
    
    return fig

@app.callback(
    Output('specific-heat-capacity-solution-figure', 'figure'),
    Input('specific-heat-capacity-dropdown', 'value'),    
    Input('specific-heat-capacity-t-slider', 'value'),
    Input('specific-heat-capacity-w-slider', 'value'),
)
def plot_specific_heat_capacity_figure(substances_list: list[str], t: list[str], w: str):
    
    def get_plot_values(name: str, _t: list[int], w: float):
        
        def get_specific_heat_capacity_solution(name, w, t, caq):
            component = specific_heat_capacity_solution[specific_heat_capacity_solution['salt'] == name].fillna(0)
            return caq - w * (component.d1 - component.d2*w - component.d3*1e-3 * t**2)
                        
        t = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= _t[0]) & (aqua_liquid_saturation['temperature'] <= _t[1])]['temperature']
        caq = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= _t[0]) & (aqua_liquid_saturation['temperature'] <= _t[1])]['specific_heat_capacity'].values
        c = [float(*get_specific_heat_capacity_solution(name, w, t_, caq_)) for t_, caq_ in list(zip(t,caq))]
        
        return pd.DataFrame({'temperature':list(t),
                             'density':c})
    
    temperature = list(map(lambda x: int(x), t))
    w = float(*w)
    
    fig = go.Figure()
    for substance in substances_list:        
        df = get_plot_values(substance, temperature, w)        
        fig.add_trace(go.Scatter(x=df['temperature'], y=df['density'], name=substance))
    
    fig.update_xaxes(title_text='Температура, °С',
                            gridcolor='rgb(105,105,105)',
                            griddash='1px',
                            zeroline=False)
    fig.update_yaxes(title_text='Дж/(кг*K)',
                    gridcolor='rgb(105,105,105)',
                    griddash='1px',
                    zeroline=False)

    #настраиваем график снаружи и на границах
    fig.update_xaxes(range=[temperature[0], temperature[1]+1],
                    showline=True, linewidth=2, linecolor='black',
                    mirror=True,
                    ticks='inside')
    fig.update_yaxes(
                    showline=True, linewidth=2, linecolor='black',
                    mirror=True,
                    ticks='inside')
    fig.update_layout(
                    autosize=False,
                    margin=dict(l=20, r=5, t=20, b=2),
                    showlegend=False,
                    plot_bgcolor='white')
    
    return fig

@app.callback(
    Output('thermal-conductivity-solution-figure', 'figure'),
    Input('thermal-conductivity-dropdown', 'value'),    
    Input('thermal-conductivity-t-slider', 'value'),
    Input('thermal-conductivity-w-slider', 'value'),
)
def plot_thermal_conductivity_figure(substances_list: list[str], t: list[str], w: str):
    
    def get_plot_values(name: str, _t: list[int], w: float):
        
        def get_thermal_conductivity_solutions(name, w, t, lyaAQ):
            component = thermal_conductivity_solutions[thermal_conductivity_solutions['salt'] == name].fillna(0)
            return lyaAQ * (1 - component.f*w)
                       
        t = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= _t[0]) & (aqua_liquid_saturation['temperature'] <= _t[1])]['temperature']
        caq = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= _t[0]) & (aqua_liquid_saturation['temperature'] <= _t[1])]['thermal_conductivity'].values
        lya = [float(*get_thermal_conductivity_solutions(name, w, t_, lyaaq_)) for t_, lyaaq_ in list(zip(t,caq))]
        
        return pd.DataFrame({'temperature':list(t),
                             'thermal_conductivity':lya})
    
    temperature = list(map(lambda x: int(x), t))
    w = float(*w)
    
    fig = go.Figure()
    for substance in substances_list:        
        df = get_plot_values(substance, temperature, w)        
        fig.add_trace(go.Scatter(x=df['temperature'], y=df['thermal_conductivity'], name=substance))
    
    fig.update_xaxes(title_text='Температура, °С',
                            gridcolor='rgb(105,105,105)',
                            griddash='1px',
                            zeroline=False)
    fig.update_yaxes(title_text='Вт/(м*K)',
                    gridcolor='rgb(105,105,105)',
                    griddash='1px',
                    zeroline=False)

    #настраиваем график снаружи и на границах
    fig.update_xaxes(range=[temperature[0], temperature[1]+1],
                    showline=True, linewidth=2, linecolor='black',
                    mirror=True,
                    ticks='inside')
    fig.update_yaxes(
                    showline=True, linewidth=2, linecolor='black',
                    mirror=True,
                    ticks='inside')
    fig.update_layout(
                    autosize=False,
                    margin=dict(l=20, r=5, t=20, b=2),
                    showlegend=False,
                    plot_bgcolor='white')
    
    return fig

@app.callback(
    Output('vicosity-solution-3d-figure', 'figure'),
    Input('vicosity-3d-dropdown', 'value'),
)
def plot_3d_vicosity(NAME):
    
    def get_plot_values(name: str, t: list[int], w: float, aq):
            
            def get_vicosity_solution(name, w, t, aq):
                component = vicosity_solution[vicosity_solution['salt'] == name].fillna(0)
                return aq*np.exp(w*(component.b1.values + component.b2.values*1e-2 * t - component.b3.values*1e-7 * t**2))
            
            u = [float(*get_vicosity_solution(name, w, t_, uaq_)) for t_, uaq_ in list(zip(t,aq))]
            return u

    t = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= 0) & (aqua_liquid_saturation['temperature'] <= 150)]['temperature']
    aq = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= 0) & (aqua_liquid_saturation['temperature'] <= 150)]['viscosity_kilo'].values/1000
    W_ = np.linspace(0,2,20)
    
    df = pd.DataFrame(index=t)
    
    for w in W_:
        df[w] = get_plot_values(str(NAME), t, w, aq)
    fig = go.Figure(data=[go.Surface(x=df.columns, y=df.index, z=df.values, colorscale='RdBu')])
    fig.update_layout(scene = dict(
                        xaxis_title='w, кг/кг',
                        yaxis_title='t, °С',
                        zaxis_title='мПа*с'),)    
    fig.update_traces(showlegend=False, selector=dict(type='surface'))
    
    return fig

@app.callback(
    Output('density-solution-3d-figure', 'figure'),
    Input('density-3d-dropdown', 'value'),
)
def plot_3d_density(NAME):
    
    def get_plot_values(name: str, t: list[int], w: float, aq):
            
            def get_density_solution(name, w, t, paq):
                component = density_solution[density_solution['salt'] == name].fillna(0)            
                return paq*np.exp(w*(component.a1 + component.a2*1e-4 * t - component.a3*1e-6 * t**2))
                        
            u = [float(*get_density_solution(name, w, t_, uaq_)) for t_, uaq_ in list(zip(t,aq))]
            return u

    t = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= 0) & (aqua_liquid_saturation['temperature'] <= 150)]['temperature']
    aq = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= 0) & (aqua_liquid_saturation['temperature'] <= 150)]['density'].values
    W_ = np.linspace(0,2,20)
    
    df = pd.DataFrame(index=t)
    
    for w in W_:
        df[w] = get_plot_values(str(NAME), t, w, aq)
    fig = go.Figure(data=[go.Surface(x=df.columns, y=df.index, z=df.values, colorscale='RdBu')])
    fig.update_layout(scene = dict(
                        xaxis_title='w, кг/кг',
                        yaxis_title='t, °С',
                        zaxis_title='кг/М^3'),)
    fig.update_traces(showlegend=False, selector=dict(type='surface'))
    
    return fig

@app.callback(
    Output('specific-heat-capacity-solution-3d-figure', 'figure'),
    Input('specific-heat-capacity-3d-dropdown', 'value'),
)
def plot_3d_specific_heat_capacity(NAME):
    
    def get_plot_values(name: str, t: list[int], w: float, aq):
            
            def get_specific_heat_capacity_solution(name, w, t, caq):
                component = specific_heat_capacity_solution[specific_heat_capacity_solution['salt'] == name].fillna(0)
                return caq - w * (component.d1 - component.d2*w - component.d3*1e-3 * t**2)
                       
            u = [float(*get_specific_heat_capacity_solution(name, w, t_, uaq_)) for t_, uaq_ in list(zip(t,aq))]
            return u

    t = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= 0) & (aqua_liquid_saturation['temperature'] <= 150)]['temperature']
    aq = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= 0) & (aqua_liquid_saturation['temperature'] <= 150)]['specific_heat_capacity'].values
    W_ = np.linspace(0,2,20)
    
    df = pd.DataFrame(index=t)
    
    for w in W_:
        df[w] = get_plot_values(str(NAME), t, w, aq)
    fig = go.Figure(data=[go.Surface(x=df.columns, y=df.index, z=df.values, colorscale='RdBu')])
    fig.update_layout(scene = dict(
                        xaxis_title='w, кг/кг',
                        yaxis_title='t, °С',
                        zaxis_title='Дж/(кг*K)'),)
    fig.update_traces(showlegend=False, selector=dict(type='surface'))
    
    return fig

@app.callback(
    Output('thermal-conductivity-solution-3d-figure', 'figure'),
    Input('thermal-conductivity-3d-dropdown', 'value'),
)
def plot_3d_thermal_conductivity(NAME):
    
    def get_plot_values(name: str, t: list[int], w: float, aq):
            
            def get_thermal_conductivity_solutions(name, w, t, lyaAQ):
                component = thermal_conductivity_solutions[thermal_conductivity_solutions['salt'] == name].fillna(0)
                return lyaAQ * (1 - component.f*w)
                                   
            u = [float(*get_thermal_conductivity_solutions(name, w, t_, uaq_)) for t_, uaq_ in list(zip(t,aq))]
            return u

    t = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= 0) & (aqua_liquid_saturation['temperature'] <= 150)]['temperature']
    aq = aqua_liquid_saturation[(aqua_liquid_saturation['temperature'] >= 0) & (aqua_liquid_saturation['temperature'] <= 150)]['thermal_conductivity'].values
    W_ = np.linspace(0,2,20)
    
    df = pd.DataFrame(index=t)
    
    for w in W_:
        df[w] = get_plot_values(str(NAME), t, w, aq)
    fig = go.Figure(data=[go.Surface(x=df.columns, y=df.index, z=df.values, colorscale='RdBu')])
    fig.update_layout(scene = dict(
                        xaxis_title='w, кг/кг',
                        yaxis_title='t, °С',
                        zaxis_title='Вт/(м*K)'),)
    fig.update_traces(showlegend=False, selector=dict(type='surface'))
    
    return fig

