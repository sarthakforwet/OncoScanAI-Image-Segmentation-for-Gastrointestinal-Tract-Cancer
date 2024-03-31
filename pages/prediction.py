import dash
from dash import html, Dash, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import cv2
import plotly.express as px
import base64
import io
from PIL import Image
from utils import build_model, load_model, CFG, plot_single
import torch
import torchvision.transforms.functional as F
from torch import nn

dash.register_page(__name__)

# Load data
df = pd.read_csv('train_data_processed.csv')

# Define page title and dropdown options
pagetitle = dcc.Markdown('# OncoScan AI: Advanced Imaging for Stomach and Intestine Cancer')
graphtitle = dcc.Markdown(children='')
case_dropdown = dcc.Dropdown(options=sorted(df['case'].unique()), value=123, clearable=False)

# Define layout
layout = dbc.Container([
    dbc.Row([
        dbc.Col([pagetitle], width=6)
    ], justify='center'),
    dbc.Row([
        dbc.Col([graphtitle], width=6)
    ], justify='left'),
    dbc.Row([
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and drop or click to select a file to upload."]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
        html.Img(id='pred_fig')
    ]),
], fluid=True)

# Callback to update prediction
@callback(
    Output('pred_fig', 'src'),
    Output(graphtitle, 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)

def update_animation(img, filenames, last_modified):
    '''
    Update the prediction animation based on the uploaded image.

    Parameters:
        img (str): Contents of the uploaded image.
        filenames (str): Filename of the uploaded image.
        last_modified (int): Last modified timestamp of the uploaded image.

    Returns:
        str: Base64-encoded image for displaying in HTML.
        str: Text for the prediction window.
    '''
    # Decode the uploaded image
    content_type, content_string = img.split(',')
    image_data = base64.b64decode(content_string)
    img = np.array(Image.open(io.BytesIO(image_data)))

    # Preprocess the image
    img = (img - img.min()) / (img.max() - img.min()) * 255.0 
    img = cv2.resize(img, (224,224))
    img = np.tile(img[...,None], [1, 1, 3])  # Convert grayscale to RGB
    img = img.astype(np.float32) / 255.
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img)
    img = img.to(CFG.device)

    # Load the model
    model = load_model(r"S:\DS 5500 - Capstone\Image-Segmentation-for-Gastrointestinal-Tract-Cancer\full_custom_model.pt", False)
    
    # Perform prediction
    with torch.no_grad():
        pred = model(img)
        pred = (nn.Sigmoid()(pred) > 0.5).double()

    # Plot the prediction on the image
    out = plot_single(img, pred)
    
    return out, 'Prediction Window'

