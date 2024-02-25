import dash
from dash import html, Dash, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import cv2
import plotly.express as px
import base64
import io
import cv2
from imageio import imread
import io
from PIL import Image
from utils import build_model, load_model, visualize_batch
import torch
import torchvision.transforms.functional as F


from torch import nn
dash.register_page(__name__)

df = pd.read_csv('train_data_processed.csv')

pagetitle = dcc.Markdown('# OncoScan AI: Advanced Imaging for Stomach and Intestine Cancer')
graphtitle = dcc.Markdown(children='')
prediction_figure = dcc.Graph(figure = {})
case_dropdown = dcc.Dropdown(options = sorted(df['case'].unique()),
                             value=123,
                             clearable=False)


layout = dbc.Container([
    dbc.Row([
        dbc.Col([pagetitle], width=6)
], justify='center'),
    dbc.Row([
        dbc.Col([graphtitle], width=6)
    ], justify='left'),
    dbc.Row([
        # dbc.Col([upload_btn], width=6),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and drop or click to select a file to upload."]
            ),
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
        dbc.Col([prediction_figure], width=6)
    ]),
], fluid=True)

@callback(
    Output(prediction_figure, 'figure'),
    Output(graphtitle, 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')

)
def update_animation(img, filenames, last_modified):
    content_type, content_string = img.split(',')
    image_data = base64.b64decode(content_string)
    image = np.array(Image.open(io.BytesIO(image_data)))
    image = np.float32(image)
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    # fig = px.imshow(image)
    # image = image.astype('float')
    image = torch.tensor(image)    
    # image = F.resize(image, (224,224,3))
    image = image.resize_(224,224,3)
    image = image.permute(2,1,0)
    image = image[None, :, :, :]


    model = load_model("vgg19-dcbb9e9d.pth")

    img = cv2.imread('pred.png', cv2.IMREAD_UNCHANGED)
    fig = px.imshow(img)
    return fig, 'Prediction Window'
    # model = torch.load_state_dict(torch.load('vgg19-dcbb9e9d.pth'))
    # model.load_state_dict(model)
    # print(model)

    with torch.no_grad():
        pred = model(image)
        pred = (nn.Sigmoid()(pred)>0.5).double()
    
    image = torch.squeeze(image, 0).permute(1,2,0)
    
    image  = image.cpu().detach()
    # preds = torch.mean(torch.stack(pred, dim=0), dim=0).cpu().detach()
    preds = [pred]
    preds = torch.mean(torch.stack(preds, dim=0), dim=0).cpu().detach()
    # print(preds.shape)
    # pred = pred.cpu().detach()
    preds = preds.resize_(224,224,1)



    fig = visualize_batch(image, preds)
    return fig, 'Prediction Window'
