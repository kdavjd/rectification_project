from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from app import app

start_tab_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Img(src='https://github.com/kdavjd/salt_hydrolysis/raw/main/images/figure_9.gif', alt="Анимированное изображение")),
        dbc.Col(html.P('Your text goes here.'))
    ]),
    dbc.Row([
        dbc.Col(html.Img(src='https://disk.yandex.ru/i/yxUEDKKWW9XjmA', alt="изображение")),
        dbc.Col(html.P('Your text goes here.'))
    ]),
    dbc.Row([
        dbc.Col(html.Img(src='your_image_url', height='200px')),
        dbc.Col(html.P('Your text goes here.'))
    ]),
], fluid=True)