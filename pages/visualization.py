import dash
from dash import html, Dash, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import cv2
import plotly.express as px

dash.register_page(__name__)

# app = Dash(__name__, external_stylesheets = [dbc.themes.VAPOR])

df = pd.read_csv('train_data_processed.csv')

pagetitle = dcc.Markdown('# OncoScan AI: Advanced Imaging for Stomach and Intestine Cancer')
graphtitle = dcc.Markdown(children='')
# animated_figure = dcc.Graph(figure = {})
case_dropdown = dcc.Dropdown(options = sorted(df['case'].unique()),
                             value=123,
                             clearable=False)

day_dropdown = dcc.Dropdown(options = sorted(df[df['case']==123]['day'].unique()),
                             value=20,
                             clearable=False)

slice_dropdown = dcc.Dropdown(options = sorted(df[(df['case']==123)&(df['day']==20)]['slice_no'].unique()),
                             value=5,
                             clearable=False)

case_mk = dcc.Markdown('Case')
day_mk = dcc.Markdown('Day')
slice_mk = dcc.Markdown('Slice Number')


viz_figure = dcc.Graph(figure={})

layout = dbc.Container([
    dbc.Row([
        dbc.Col([pagetitle], width=6)
], justify='center'),
    dbc.Row([
        dbc.Col([graphtitle], width=6)
    ], justify='left'),
    dbc.Row([
        dbc.Col([viz_figure], width=9)
    ], justify='center'),
    dbc.Row([
        dbc.Col([case_mk], width=3),
        dbc.Col([day_mk], width=3),
        dbc.Col([slice_mk], width=3)
    ], justify='center'),
    dbc.Row([
        dbc.Col([case_dropdown], width=3),
        dbc.Col([day_dropdown], width=3),
        dbc.Col([slice_dropdown], width=3),
    ], justify='center')
], fluid=True)

@callback(
    Output(day_dropdown, 'options'),
    Input(case_dropdown, 'value')
)
def output_day(case):
    days = df.loc[df['case']==case]['day'].unique()
    day_dropdown = dcc.Dropdown(options = sorted(days),
                             value=20,
                             clearable=False)
    return days

@callback(
    Output(slice_dropdown, 'options'),
    Input(case_dropdown, 'value'),
    Input(day_dropdown, 'value')
)
def output_slice(case, day):
    slices = df.loc[(df['case']==case)&(df['day']==day)]['slice_no'].unique()
    return slices

@callback(
    Output(viz_figure, 'figure'),
    Input(case_dropdown, 'value'),
    Input(day_dropdown, 'value'),
    Input(slice_dropdown, 'value')
)
def update_visualization(case, day, slice):

    path = df.loc[(df['case']==case)&(df['day']==day)&(df['slice_no']==slice)]['path'].values[1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    fig = px.imshow(img)
    return fig
