from dash import Dash, Input, Output, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import cv2
from utils import get_id_mask

df = pd.read_csv('train_data_processed.csv')

app = Dash(__name__, external_stylesheets = [dbc.themes.VAPOR])

pagetitle = dcc.Markdown('# OncoScan AI: Advanced Imaging for Stomach and Intestine Cancer')
graphtitle = dcc.Markdown(children='')
main_dropdown = dcc.Dropdown(options = ['Visualize through Case/Day/Slice', 'Animated Segmentations', 'Predict Segmentation'],
                        value='Animated Segmentations',
                        clearable=False)

animated_figure = dcc.Graph(figure = {})
case_dropdown = dcc.Dropdown(options = sorted(df['case'].unique()),
                             value=123,
                             clearable=False)


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([pagetitle], width=6)
], justify='center'),
    dbc.Row([
        dbc.Col([graphtitle], width=6)
    ], justify='left'),
    dbc.Row([
        dbc.Col([main_dropdown], width=3),
        dbc.Col([animated_figure], width=9)
    ]),
    dbc.Row([
        dbc.Col([case_dropdown], width=6)
    ], justify='center')
], fluid=True)

@app.callback(
    Output(animated_figure, 'figure'),
    Output(graphtitle, 'children'),
    Input(main_dropdown, 'value'),
    Input(case_dropdown, 'value')
)
def update_animation(type, case):
    if type=='Animated Segmentations':
        img_paths = df.loc[df['case']==case][['path', 'ids']]
        masks = []
        for path, id in zip(img_paths, img_ids):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            # Resize the image to 224 x 224.

            mask = get_id_mask(id, df)
            masks.append(mask)

        imgs = np.array(imgs)
        fig = px.imshow(imgs, animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
    return fig, 'Animated Window'

if __name__=="__main__":
    app.run_server(port=8051)