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

pagetitle = dcc.Markdown('Please allow some time to load...')
graphtitle = dcc.Markdown(children='')
animated_figure = dcc.Graph(figure = {})
case_dropdown = dcc.Dropdown(options = sorted(df['case'].unique()),
                             value=123,
                             clearable=False)

layout = dbc.Container([
    dbc.Row([
        dbc.Col([pagetitle], width=12)
], justify='left'),
    dbc.Row([
        dbc.Col([graphtitle], width=6)
    ], justify='left'),
    dbc.Row([
        dbc.Col([
            dcc.Loading(
            id='loading',
            type='circle',
            children = html.Div(id = 'loading_output'))], style={'height':'1.8cm'})
            ]),
    dbc.Row([
        dbc.Col([animated_figure], width=9)
    ]),
    dbc.Row([
        dbc.Col([case_dropdown], width=6),
    ], justify='center')
], fluid=True)

@callback(
    Output('loading_output', 'children'),
    Output(animated_figure, 'figure'),
    Output(graphtitle, 'children'),
    Input(case_dropdown, 'value'),
)
def update_animation(case):
    case_df = df.loc[(df['case']==case)]
    idxs = case_df.index
    img_paths = case_df['path'].values
    
    masks_arr = []
    imgs = []
    figs = []
    for path, idx in zip(img_paths, idxs):            
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # # Resize the image to 224 x 224.
        img = cv2.resize(img, (224,224))
        imgs.append(img)

    imgs = np.array(imgs)
    # masks_arr = np.array(masks_arr)

    fig = px.imshow(imgs, animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
    # fig.add_heatmap(z=masks, opacity=0.5, colorscale='Viridis', zmin=0, zmax=1)

    return  '', fig, 'Animation Loaded Successfully'