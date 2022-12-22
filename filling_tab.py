import glob
import pandas as pd
import numpy as np
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from app import app

from column_functions import Calculations as clc
from column_functions import Figures as figures
from data_functions import DataFunctions as dfс
from equipment_functions import Calculations as eq

am = pd.read_excel('data/atomic_mass.xlsx')
aqua_liquid_saturation = pd.read_excel('data/ph_H2O_saturation_liquid.xlsx')
aqua_vapor_saturation = pd.read_excel('data/ph_H2O_saturation_gas.xlsx')
aqua_vapor_saturation_by_pressure = pd.read_excel('data/ph_H2O_saturation_gas_by_pressure.xlsx')
mass_concentration_solution = pd.read_excel('data/mass_concentration_solution.xlsx')
boiling_point_solution = pd.read_excel('data/boiling_point_solution.xlsx')
density_solution = pd.read_excel('data/density_solution.xlsx')
vicosity_solution = pd.read_excel('data/vicosity_solution.xlsx')
specific_heat_capacity_solution = pd.read_excel('data/heat_capacity_solution.xlsx')
thermal_conductivity_solutions = pd.read_excel('data/thermal_conductivity_solutions.xlsx')
ph_gases = pd.read_excel('data/ph_gases.xlsx')
ph_organic = pd.read_excel('data/ph_organic.xlsx')
density_organic_liquid = pd.read_excel('data/density_organic_liquid.xlsx')
vicosity_organic_liquid = pd.read_excel('data/vicosity_organic_liquid.xlsx')
interfactial_tension_organic_liquid = pd.read_excel('data/interfacial_tension_organic_liquid.xlsx')
thermal_expansion_organic_liquid = pd.read_excel('data/thermal_expansion_organic_liquid.xlsx')
heat_capacity_organic_liquid = pd.read_excel('data/heat_capacity_organic_liquid.xlsx')
thermal_conductivity_organic_liquid = pd.read_excel('data/thermal_conductivity_organic_liquid.xlsx')
heat_vaporization_organic_liquid = pd.read_excel('data/heat_vaporization_organic_liquid.xlsx')
vapor_pressure_organic_liquid = pd.read_excel('data/vapor_pressure_organic_liquid.xlsx')
vicosity_organic_vapor = pd.read_excel('data/vicosity_organic_vapor.xlsx')

exclude_list =['salts','name','formula','salt']
#Поверхностное натяжение органических жидкостей [мДж/м^2]
interfactial_tension_organic_liquid = dfс.delete_hyphens(interfactial_tension_organic_liquid,exclude_list)

#Теплопроводность органических жидкостей [Вт/(м*K)]
thermal_conductivity_organic_liquid = dfс.delete_hyphens(thermal_conductivity_organic_liquid,exclude_list)

#Свойства водяного пара в состянии насыщения в зависимости от давления
aqua_vapor_saturation_by_pressure = dfс.delete_hyphens(aqua_vapor_saturation_by_pressure,exclude_list)

#Коэффициенты объемного теплового расширения органических жидкостей b*10^3, K^-1
thermal_expansion_organic_liquid = dfс.delete_hyphens(thermal_expansion_organic_liquid,exclude_list)

#Теплота парообразования органических жидкостей [кДж/кг]
heat_vaporization_organic_liquid = dfс.delete_hyphens(heat_vaporization_organic_liquid,exclude_list)

#Удельная  теплоемкость водных р-ров cp = cpAQ - w * (d1 - d2*w -d3e-3 *t^2)
specific_heat_capacity_solution = dfс.delete_hyphens(specific_heat_capacity_solution,exclude_list)

#Теплопроводность водных р-ров неорганических соединений lya = lyaAQ * (1 - f*w)
thermal_conductivity_solutions = dfс.delete_hyphens(thermal_conductivity_solutions,exclude_list)

#Давление насыщенного пара [мм.рт.ст.] над органической жидкостью
vapor_pressure_organic_liquid = dfс.delete_hyphens(vapor_pressure_organic_liquid,exclude_list)

#Удельная теплоемкость [Дж/(кг*K)] органических жидкостей
heat_capacity_organic_liquid = dfс.delete_hyphens(heat_capacity_organic_liquid,exclude_list)

#Концентрации насыщенных водных растворов неорганических веществ кг/кг при °С
mass_concentration_solution = dfс.delete_hyphens(mass_concentration_solution,exclude_list)

#Динамическая вязкость органических жидкостей [мПа*с]
vicosity_organic_liquid = dfс.delete_hyphens(vicosity_organic_liquid,exclude_list)

#Плотность органических жидкостей [кг/м^3]
density_organic_liquid = dfс.delete_hyphens(density_organic_liquid,exclude_list)

#Температуры кипения водных растворов неорганических веществ при н.у.
boiling_point_solution = dfс.delete_hyphens(boiling_point_solution,exclude_list)

#Физические свойства воды на линии насыщения
aqua_liquid_saturation = dfс.delete_hyphens(aqua_liquid_saturation,exclude_list)

#Вязкость паров органических веществ [мкПа*с]
vicosity_organic_vapor = dfс.delete_hyphens(vicosity_organic_vapor,exclude_list)

#Свойства водяного пара в состянии насыщения в зависимости от температуры
aqua_vapor_saturation = dfс.delete_hyphens(aqua_vapor_saturation,exclude_list)

#Вязкость водных растворов неорганических веществ u = uaq*exp^[w(b1 + b2e-2 * t - b3e-7 * t^2)]
vicosity_solution = dfс.delete_hyphens(vicosity_solution,exclude_list)

#Плотность водных растворов p = paq*exp^[w*(a1 + a2e-4 * t - a3e-6 * t^2)], w[кг/кг], t[°C], p[кг/м^3], u[Па*c]
density_solution = dfс.delete_hyphens(density_solution,exclude_list)

#Основные характеристики органических веществ
ph_organic = dfс.delete_hyphens(ph_organic,exclude_list)

#Свойства газов при н.у.
ph_gases = dfс.delete_hyphens(ph_gases,exclude_list)
#am = delete_hyphens(am)

class Сomponent():
                
    def __init__(self, name):        
        self.interfactial_tension_organic_liquid = interfactial_tension_organic_liquid[interfactial_tension_organic_liquid['name'] == name].drop('name', axis=1)
        self.thermal_conductivity_organic_liquid = thermal_conductivity_organic_liquid[thermal_conductivity_organic_liquid['name'] == name].drop('name', axis=1)
        self.thermal_expansion_organic_liquid = thermal_expansion_organic_liquid[thermal_expansion_organic_liquid['name'] == name].drop('name', axis=1)
        self.heat_vaporization_organic_liquid = heat_vaporization_organic_liquid[heat_vaporization_organic_liquid['name'] == name].drop('name', axis=1)        
        self.vapor_pressure_organic_liquid = vapor_pressure_organic_liquid[vapor_pressure_organic_liquid['name'] == name].drop('name', axis=1)
        self.heat_capacity_organic_liquid = heat_capacity_organic_liquid[heat_capacity_organic_liquid['name'] == name].drop('name', axis=1)
        self.vicosity_organic_liquid = vicosity_organic_liquid[vicosity_organic_liquid['name'] == name].drop('name', axis=1)
        self.vicosity_organic_liquid = vicosity_organic_liquid[vicosity_organic_liquid['name'] == name].drop('name', axis=1)
        self.density_organic_liquid = density_organic_liquid[density_organic_liquid['name'] == name].drop('name', axis=1)
        self.vicosity_organic_vapor = vicosity_organic_vapor[vicosity_organic_vapor['name'] == name].drop('name', axis=1)
        self.ph_organic = ph_organic[ph_organic['name'] == name]

def get_all_diagrams():
    def filter_diagrams(diagram):
        exclude_list = ['H2O', 'H2O_p', 'HCl']
        for substance in exclude_list:
            if substance in diagram:
                return False
        return True

    file_list = [file_name[:-5] for file_name in glob.glob('*.xlsx', root_dir='l_v')]
    diagrams_list = [[file_name[0:file_name.find('-')],file_name[file_name.find('-')+1:]] for file_name in file_list]
    diagrams = list(filter(filter_diagrams, diagrams_list))

    return diagrams
diagrams = get_all_diagrams()

def get_diagrams_options():
    diagrams_labels = []
    diagrams_values = []
    for diagram in diagrams:
        diagrams_values.append(
            diagram[0] + '-' + diagram[1])
        diagrams_labels.append(
            str(*ph_organic[ph_organic['formula'] == diagram[0]]['name'].values)
            +'-'
            +str(*ph_organic[ph_organic['formula'] == diagram[1]]['name'].values))
    return diagrams_labels, diagrams_values
diagrams_labels, diagrams_values = get_diagrams_options()

def get_a_name(name):
    return name[0:name.find('-')]

def get_b_name(name):
    return name[name.find('-')+1:]

Substance = {'A':Сomponent(name='Метанол'), 'B':Сomponent(name='Этанол')}
A_name = Substance['A'].ph_organic['formula'].values
B_name = Substance['B'].ph_organic['formula'].values

diagram = pd.read_excel('l_v/CH3OH-CH3CH2OH.xlsx')

if diagram['x'].values.max() > 1:
    diagram['x'] = diagram['x']/100
    
if diagram['y'].values.max() > 1:
    diagram['y'] = diagram['y']/100
    
diagram.sort_values(by = ['t'], ascending=False,ignore_index=True, inplace=True)
xy_diagram = dfс.get_coeffs(diagram['x'], diagram['y'])



F = np.double(5)                  #Производительность по исходной смеси кг/с
FEED_TEMPERATURE = np.double(20)  #Начальная температура
FEED = np.double(0.35)            #В исходной смеси %масс Ллт 
DISTILLATE = np.double(0.98)      #В дистилляте(ректификате) %масс 
BOTTOM = np.double(0.017)         #В кубовом остатке %масс ллт
PRESSURE = np.double(10**5)       #Давление в колонне в Па. Влияет на коэфф. диффузии пара в колонне


balance = clc.material_balance(F, FEED, DISTILLATE, BOTTOM, xy_diagram, Substance)
balance.apply(lambda x: np.round(x,2))

phlegm_number_fig, R, Ngraf = clc.get_range_phlegm_number(
    balance['yf'],
    balance['xw'],
    balance['xf'],
    balance['xp'],
    balance['Rmin'],
    xy_diagram,
    diagram,
    Bt_range=20,#изменяемый параметр
    plot_type='plotly')

Ropt_fig, Ropt = clc.get_optimal_phlegm_number(R, Ngraf, plot_type='plotly')
properties = clc.calculate_properties(diagram, balance, Substance)
transfer_numbers_fig, bottom, top = clc.get_transfer_numbers(balance, Ropt, xy_diagram, plot_type='plotly')
diameter = clc.calculate_diameter(balance, Ropt, properties, filling_name='50x50x5')

height = clc.calculate_hight(
    balance,
    properties,
    diameter,
    xy_diagram,
    bottom,
    top,
    Substance,
    Ropt,
    PRESSURE,
    filling_name='50x50x5')

thermal_balance = clc.calculate_thermal_balance(balance, properties, Ropt)

#таблица


#выпадающие списки
properties_dropdown = dcc.Dropdown(
    id='properties-dropdown',
    options=[{'label':column, 'value':column} for column in properties.columns],
    value=list(properties.columns[0:6]),
    multi=True)

diagrams_dropdown = dcc.Dropdown(
    id='diagrams-dropdown',
    options=[{'label':label, 'value':value} for label,value in list(zip(diagrams_labels, diagrams_values))],
    value=diagrams_values[0])

#импуты
inputs = html.Div(
    [
        html.Div('Исходные данные на проектирование'),
        dbc.Input(id='F', placeholder="Производительность, кг/с", size="sm"),
        dbc.Input(id='FEED-TEMPERATURE', placeholder="Температура смеси, °С", size="sm"),
        dbc.Input(id='FEED', placeholder="доля ЛЛТ в исходной смеси", size="sm"),
        dbc.Input(id='DISTILLATE', placeholder="доля ЛЛТ в дистилляте", size="sm"),
        dbc.Input(id='BOTTOM', placeholder="доля ЛЛТ в кубе", size="sm"),
        dbc.Input(id='PRESSURE', placeholder="давление внутри колонны, Па (10**5)", size="sm"),
    ])

inline_radioitems = html.Div(
    [
        dbc.Label("Выберите размер насадки"),
        dbc.RadioItems(
            options=[
                {"label": "50x50x5", "value": '50x50x5'},
                {"label": '35x35x4', "value": '35x35x4'},
                {"label": '25x25x3', "value": '25x25x3'},
            ],
            value=1,
            id="filling-input",
            inline=True
            )])

button = html.Div([dbc.Button("Выполнить расчет", size="lg", id='main-button')])

filling_layout = html.Div([    
    dbc.Row([dbc.Col([html.Div('Выберите бинарную смесь '),
                      html.Div([diagrams_dropdown, inputs, inline_radioitems, html.Hr(), button])], width=3),
            dbc.Col([html.Div(id='diagram-table')], width={"size": 2, "offset": 0}),
            dbc.Col([dcc.Graph(id='diagram-figure')])]),
    html.Hr(),
    dbc.Row([dbc.Col([html.Div(id='balance-table')], width={"size": 3, "offset": 0}),
             dbc.Col([html.Div('Выберите физико-химические свойства веществ для таблицы ниже: '),
                      html.Div(properties_dropdown),
                      html.Div(id='properties-table')], width=9)]),
    dbc.Row([dbc.Col([dcc.Graph(id='range-phlegm-number-figure')])]),    
    ],
    style={'margin-left': '10px',
           'margin-right': '10px'}
)

@app.callback(
    Output(component_id='properties-table', component_property='children'),
    Input(component_id='properties-dropdown', component_property='value')
)
def create_preperties_table(properties_list):
    return dbc.Table.from_dataframe(round(properties[properties_list], 3), index=True)

@app.callback(
    Output(component_id='diagram-table', component_property='children'),
    Output(component_id='diagram-figure', component_property='figure'),
    Input(component_id='diagrams-dropdown', component_property='value')
)
def get_diagram(substances):
    
    global diagram
    global Substance
    
    def ends(df, x=5):
        return pd.concat([df.head(x), df.tail(x)])
    
    path = 'l_v/'+substances+'.xlsx'
    diagram = pd.read_excel(path)

    if diagram['x'].values.max() > 1:
        diagram['x'] = diagram['x']/100
        
    if diagram['y'].values.max() > 1:
        diagram['y'] = diagram['y']/100
        
    diagram.sort_values(by = ['t'], ascending=False,ignore_index=True, inplace=True)
    
    Substance = {'A':Сomponent(name=str(*ph_organic[ph_organic.formula == get_a_name(substances)].name.values)),
                 'B':Сomponent(name=str(*ph_organic[ph_organic.formula == get_b_name(substances)].name.values))}
    
    print(Substance['A'].ph_organic.name) 
    
    return (html.Div(dbc.Table.from_dataframe(df=round(ends(diagram),2), index=True)),
            figures.plot_xy_diagram(diagram, get_a_name(substances), plot_type='plotly'))
    
@app.callback(
    Output("balance-table", "children"),
    Output("range-phlegm-number-figure", "figure"),    
    [State("F", "value"),
     State("FEED-TEMPERATURE", "value"),
     State("FEED", "value"),
     State("DISTILLATE", "value"),
     State("BOTTOM", "value"),
     State("PRESSURE", "value"),
     Input("main-button", "n_clicks"),
     ]
)
def on_button_click(F, FEED_TEMPERATURE, FEED, DISTILLATE, BOTTOM, PRESSURE, BUTTON):
    
    global balance        
    global Substance
    global xy_diagram
    global balance
    global properties
    
    if BUTTON == 0:
        return(dbc.Table.from_dataframe(balance, index=True, header=False),
                phlegm_number_fig)
    else:
        
        if F == None:
            F = np.double(5)
        else:
            F=np.double(F)
            
        if FEED_TEMPERATURE == None:
            FEED_TEMPERATURE = np.double(20)
        else:
            FEED_TEMPERATURE = np.double(FEED_TEMPERATURE)
            
        if FEED == None:
            FEED = np.double(0.35)
        else:
            FEED = np.double(FEED)
            
        if DISTILLATE == None:
            DISTILLATE = np.double(0.98)
        else:
            DISTILLATE = np.double(DISTILLATE)
        
        if BOTTOM == None:
            BOTTOM = np.double(0.03)
        else:
            BOTTOM = np.double(BOTTOM)
            
        if PRESSURE == None:
            PRESSURE = np.double(10**5)
        else:
            PRESSURE = np.double(PRESSURE)

                
        xy_diagram = dfс.get_coeffs(diagram['x'], diagram['y'])
        
        balance = clc.material_balance(F, FEED, DISTILLATE, BOTTOM, xy_diagram, Substance)
        balance = balance.apply(lambda x: np.round(x,2))
        
        properties = clc.calculate_properties(diagram, balance, Substance)
        
        
        phlegm_number_fig, R, Ngraf = clc.get_range_phlegm_number(
            balance['yf'],
            balance['xw'],
            balance['xf'],
            balance['xp'],
            balance['Rmin'],
            xy_diagram,
            diagram,
            Bt_range=20,#изменяемый параметр
            plot_type='plotly')
        
        
        return (dbc.Table.from_dataframe(balance, index=True, header=False),
                phlegm_number_fig,
                )