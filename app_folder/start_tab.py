from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from app_folder.app import app

links_style = {'display': 'block',
               'margin': '15px',
               'line-height': '2.5',
               }

start_tab_layout = dbc.Container([
    dbc.Row([
        dbc.Col([            
            html.H4("Найти меня можно тут: "),
            html.Div(html.A("Вконтакте", href="https://vk.com/davjd"), style=links_style),
            html.Div(html.A("Телеграм", href="https://t.me/NuclearExistence"), style=links_style),#Солевой гидролиз хлорида титана.
            html.Div(html.A("github", href="https://github.com/kdavjd"), style=links_style),
        ], width={"size": 3, "offset": 0}), 
        dbc.Col([            
            html.P("Во вкладках находятся тренажеры по изучению процесса ректификации. Ниже я выкладываю ссылки на репозитории своих работ, которые не пошли в публикацию в журналы по разным причинам. Пожалуйста, не бросайте вызов моим навыкам программирования, когда считаете колонны. Сломать расчет некорректными данными действительно возможно. Если вдруг в своих исканиях вы дошли до того что сервер перестал отвечать, напишите мне пожалуйста и я его перезагружу. Либо подождите немного, раз в пару дней я сюда захожу. Сервер у меня слабый, по этому при открытии вкладок и при расчете теплообменников придется подождать несколько секунд. "),
        ], width={"size": 6, "offset": 0}), 
        dbc.Col([
            html.Img(src="https://via.placeholder.com/250"),
        ], width={"size": 3, "offset": 0})
             ]),
    html.Hr(),
    dbc.Row([dbc.Col(html.A("Солевой гидролиз хлорида титана.", href="https://github.com/kdavjd/salt_hydrolysis"), 
                     width={"size": 2, "offset": 0}), 
             dbc.Col(html.P('Работа о том как я решал задачу повышения химической активности оксида титана и попутно обнаружил удивительной красоты феномены. К сожалению, дальнейшее исследование выходило за рамки целей работы и полноты экспериментальной базы не хватило для того чтобы углубиться в мои гипотезы. Таким образом, работа носит спекулятивный характер. '),
                     width={"size": 9, "offset": 1})]),
    dbc.Row([
        dbc.Col(dcc.Markdown(r'$\text{Rutile} \xrightarrow{\gamma_1} \text{Brukite} \xrightarrow{\gamma_2} \text{Anatase}$', mathjax=True),
                width={"size": 4, "offset": 2}),
        dbc.Col(dcc.Markdown('Откуда:'),
                width={"size": 3, "offset": 0}),
        dbc.Col(dcc.Markdown('Выражая через концентрацию рутила, приведённую к ст. условиям:'),
                width={"size": 3, "offset": 0}),
        ]),
    dbc.Row([        
        dbc.Col(html.Img(src='https://github.com/kdavjd/salt_hydrolysis/raw/main/images/figure_9.gif', 
                         alt="Анимированное изображение",
                         height='320px',
                         width='660px'), 
                width={"size": 5, "offset": 0}),
        dbc.Col([
            dcc.Markdown(r'$\frac{dRut}{dt}= -\gamma_1 Rut$', mathjax=True),
            dcc.Markdown(r'$\frac{dBr}{dt}= \gamma_1 Rut-\gamma_2 Br$', mathjax=True),
            dcc.Markdown(r'$\frac{dAn}{dt}= \gamma_2 Br$', mathjax=True),            
            dcc.Markdown(r'$A= \frac{\gamma_{1}}{\gamma_{st}}$', mathjax=True),
            dcc.Markdown(r'$B= \frac{\gamma_{2}}{\gamma_{1}}$', mathjax=True),],
            width={"size": 2, "offset": 2}),
        dbc.Col([
            dcc.Markdown(r'$Rut= Rut_{0}e^{-\gamma_{1}t}$', mathjax=True),
            dcc.Markdown(r'$Br= \frac{Rut_{0}(1 - e^{-\gamma_{1}t})}{\gamma_{1}} \cdot (1 - e^{-\gamma_{2}(t - t_{1})})$', mathjax=True),
            dcc.Markdown(r'$An= \frac{Rut_{0}(1 - e^{-\gamma_{1}t})}{\gamma_{1}} \cdot e^{-\gamma_{2}(t - t_{1})}$', mathjax=True),],
                width={"size": 3, "offset": 0}),
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col(),
        dbc.Col(),
    ]),
    dbc.Row([
        dbc.Col(),
        dbc.Col()
    ]),
], fluid=True)

