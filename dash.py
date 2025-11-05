import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import io
import base64

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("🏏 Cricket Analysis Dashboard"),
    html.Hr(),

    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag & Drop or ', html.A('Select a CSV File')]),
        style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
               'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
               'textAlign': 'center', 'margin-bottom': '10px'},
        multiple=False
    ),

    dbc.Row([
        dbc.Col(dcc.Dropdown(id='bat-team', placeholder="Select Batting Team"), width=4),
        dbc.Col(dcc.Dropdown(id='batsman', placeholder="Select Batsman"), width=4),
        dbc.Col(dcc.Dropdown(id='delivery', placeholder="Select Delivery Type"), width=4),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id='crease-beehive'), width=6),
        dbc.Col(dcc.Graph(id='zonal-boxes'), width=6),
    ]),
])

# --- CALLBACKS ---
@app.callback(
    [Output('bat-team', 'options'),
     Output('batsman', 'options'),
     Output('delivery', 'options'),
     Output('crease-beehive', 'figure'),
     Output('zonal-boxes', 'figure')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_dashboard(contents, filename):
    if contents is None:
        return [], [], [], go.Figure(), go.Figure()

    # Decode the uploaded CSV
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Simple sample charts
    fig_cbh = go.Figure(data=[go.Scatter(
        x=df['StumpsY'], y=df['StumpsZ'], mode='markers', marker=dict(color='blue')
    )])
    fig_cbh.update_layout(title="Crease Beehive")

    fig_zonal = go.Figure(data=[go.Histogram(x=df['Runs'])])
    fig_zonal.update_layout(title="Runs Distribution")

    teams = [{'label': t, 'value': t} for t in df['BattingTeam'].dropna().unique()]
    batsmen = [{'label': b, 'value': b} for b in df['BatsmanName'].dropna().unique()]
    deliveries = [{'label': d, 'value': d} for d in df['DeliveryType'].dropna().unique()]

    return teams, batsmen, deliveries, fig_cbh, fig_zonal


if __name__ == "__main__":
    app.run_server(debug=True)
