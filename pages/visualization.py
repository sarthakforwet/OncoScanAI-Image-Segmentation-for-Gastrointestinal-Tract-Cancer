import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import cv2
import plotly.express as px

# Register the current module as a Dash page
dash.register_page(__name__)

# Read the dataset
df = pd.read_csv('train_data_processed.csv')

# Define page components
pagetitle = dcc.Markdown('# OncoScan AI: Advanced Imaging for Stomach and Intestine Cancer')
graphtitle = dcc.Markdown(children='')
case_dropdown = dcc.Dropdown(options=sorted(df['case'].unique()), value=123, clearable=False)
day_dropdown = dcc.Dropdown(options=sorted(df[df['case']==123]['day'].unique()), value=20, clearable=False)
slice_dropdown = dcc.Dropdown(options=sorted(df[(df['case']==123) & (df['day']==20)]['slice_no'].unique()), value=5, clearable=False)
case_mk = dcc.Markdown('Case')
day_mk = dcc.Markdown('Day')
slice_mk = dcc.Markdown('Slice Number')
viz_figure = dcc.Graph(figure={})

# Define layout
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

# Define callbacks
@callback(
    Output(day_dropdown, 'options'),
    Input(case_dropdown, 'value')
)
def output_day(case):
    '''
    Update the options for the day dropdown based on the selected case.

    Parameters:
        case (int): Selected case.

    Returns:
        list: Updated options for the day dropdown.
    '''
    days = df.loc[df['case']==case]['day'].unique()
    return [{'label': str(day), 'value': day} for day in sorted(days)]

@callback(
    Output(slice_dropdown, 'options'),
    Input(case_dropdown, 'value'),
    Input(day_dropdown, 'value')
)
def output_slice(case, day):
    '''
    Update the options for the slice dropdown based on the selected case and day.

    Parameters:
        case (int): Selected case.
        day (int): Selected day.

    Returns:
        list: Updated options for the slice dropdown.
    '''
    slices = df.loc[(df['case']==case) & (df['day']==day)]['slice_no'].unique()
    return [{'label': str(slice), 'value': slice} for slice in sorted(slices)]

@callback(
    Output(viz_figure, 'figure'),
    Input(case_dropdown, 'value'),
    Input(day_dropdown, 'value'),
    Input(slice_dropdown, 'value')
)
def update_visualization(case, day, slice):
    '''
    Update the visualization based on the selected case, day, and slice.

    Parameters:
        case (int): Selected case.
        day (int): Selected day.
        slice (int): Selected slice.

    Returns:
        plotly.graph_objects.Figure: Updated visualization figure.
    '''
    path = df.loc[(df['case']==case) & (df['day']==day) & (df['slice_no']==slice)]['path'].values[0]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    fig = px.imshow(img)
    return fig
