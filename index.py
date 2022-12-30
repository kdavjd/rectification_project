from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

from app import app
from filling_tab import filling_layout
from plate_tab import plate_layout
from table_tab import table_tab_layout

server = app.server

app_tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(label="Насадочная колонна", tab_id="tab-filling"),
                dbc.Tab(label="Тарельчатая колонна", tab_id="tab-plate"),
                dbc.Tab(label="Таблица веществ", tab_id="table-tab-plate"),
                dbc.Tab(label="Other", tab_id="tab-other"),
            ],
            id="tabs",
            active_tab="tab-filling",
        ),
    ])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Иллюстративный расчет ректификационной колонны",
                            style={"textAlign": "center"}), width=12)),
    html.Hr(),
    dbc.Row(dbc.Col(app_tabs, width=12)),
    html.Div(id='content', children=[])])


@app.callback(
    Output("content", "children"),
    [Input("tabs", "active_tab")]
)
def switch_tab(tab_chosen):
    if tab_chosen == "tab-filling":
        return filling_layout
    elif tab_chosen == "tab-plate":
        return plate_layout
    elif tab_chosen == "table-tab-plate":
        return table_tab_layout
    elif tab_chosen == "tab-other":
        pass
    return html.P("???????")


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
