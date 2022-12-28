from dash import html
import dash_bootstrap_components as dbc

from app_data import *


table_tab_layout = html.Div([dbc.Table.from_dataframe(df=ph_organic, index=True)])