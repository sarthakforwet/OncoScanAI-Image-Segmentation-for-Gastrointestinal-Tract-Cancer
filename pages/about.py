import dash
from dash import html, Dash, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import cv2
import plotly.express as px

dash.register_page(__name__, path='/')

about = dcc.Markdown(children = "# OncoScan: Image Segmentation for Gastrointestinal Tract Cancer")

layout = html.Div([
    # html.Img(src="S:/DS 5500 - Capstone/Image-Segmentation-for-Gastrointestinal-Tract-Cancer/about_page_image.jpg", alt='My Image'),
    html.H1("Home Page", style={'textAlign':'center'}),
    html.Div([
        html.P('OncoScan AI is a platform for masking images of stomach and intestine from the tumor through analysis on our gastrointestinal MRI Scan. Our interface would allow you to get a \
               thorough understanding of how the MRI scans look like with its masked segmentation of stomach and intestine from the tumor which helps the machine to direct radiation directly on tumor \
               while avoiding the internal organs.'),
        html.H3("Visualization"),
        html.P('This page would allow to visualize a particular MRI scan image by providing the information about Case, Day and Slice Number'),
        html.P('The Days would get updated on the basis of the selected case and the same happens for the slice on a particular combination of day and case.'),
        html.H3("Animation"),
        html.P('This page would allow to visualize the MRI scans for all the slices and days for particular cases which can be selected through the dropdown button.'),
        html.H3("Prediction"),
        html.P('This page would allow a user to predict a segmentation mask in an Image using a pretrained model in the backend.'),
    ], style={'margin-top': '20px'})
])