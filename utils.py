
import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"
import random
from glob import glob
import os, shutil
from tqdm import tqdm
tqdm.pandas()
import time

from collections import defaultdict
import gc

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Sklearn
from sklearn.model_selection import StratifiedGroupKFold

# PyTorch 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

from joblib import Parallel, delayed
from matplotlib.patches import Rectangle

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import segmentation_models_pytorch as smp
import torch, torchvision.models
import plotly.express as px


class CFG:
    seed          = 101
    debug         = False # set debug=False for Full Training
    exp_name      = 'Baselinev2'
    comment       = 'unet-efficientnet_b1-224x224-aug2-split2'
    model_name    = 'Unet'
    backbone      = 'efficientnet-b1'
    train_bs      = 128
    valid_bs      = train_bs*2
    img_size      = [224, 224]
    epochs        = 15
    lr            = 2e-3
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(30000/train_bs*epochs)+50
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    n_accumulate  = max(1, 32//train_bs)
    n_fold        = 5
    num_classes   = 3
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mask_from_segmentation(segmentation, shape):
    '''Returns the mask corresponding to the inputed segmentation.
    segmentation: a list of start points and lengths in this order
    max_shape: the shape to be taken by the mask
    return:: a 2D mask'''

    # Get a list of numbers from the initial segmentation
    segm = np.asarray(segmentation.split(), dtype=int)

    # Get start point and length between points
    start_point = segm[0::2] - 1
    length_point = segm[1::2]

    # Compute the location of each endpoint
    end_point = start_point + length_point

    # Create an empty list mask the size of the original image
    # take onl
    case_mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    # Change pixels from 0 to 1 that are within the segmentation
    for start, end in zip(start_point, end_point):
        case_mask[start:end] = 1

    case_mask = case_mask.reshape((shape[0], shape[1]))
    
    return case_mask

def get_id_mask(ID, train, verbose=False):
    '''Returns a mask for each case ID. If no segmentation was found, the mask will be empty
    - meaning formed by only 0
    ID: the case ID from the train.csv file
    verbose: True if we want any prints
    return: segmentation mask'''

    # ~~~ Get the data ~~~
    # Get the portion of dataframe where we have ONLY the speciffied ID
    ID_data = train[train["id"]==ID].reset_index(drop=True)

    # Split the dataframe into 3 series of observations
    # each for one speciffic class - "large_bowel", "small_bowel", "stomach"
    observations = [ID_data.loc[k, :] for k in range(3)]

    # ~~~ Create the mask ~~~
    # Get the maximum height out of all observations
    # if max == 0 then no class has a segmentation
    # otherwise we keep the length of the mask
    max_height = np.max([obs.image_height for obs in observations])
    max_width = np.max([obs.image_width for obs in observations])

    # Get shape of the image
    # 3 channels of color/classes
    shape = (max_height, max_width, 3)

    # Create an empty mask with the shape of the image
    mask = np.zeros(shape, dtype=np.uint8)

    # If there is at least 1 segmentation found in the group of 3 classes
    if max_height != 0:
        for k, location in enumerate(["large_bowel", "small_bowel", "stomach"]):
            observation = observations[k]
            segmentation = observation.segmentation

            # If a segmentation is found
            # Append a new channel to the mask
            if pd.isnull(segmentation) == False:
                mask[..., k] = mask_from_segmentation(segmentation, shape)
            
    return mask

def rle_decode(mask_rle, shape):
    '''
    Function to Perform Run Length Decoding
    
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:-1][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


# Function to decode run-length encoding
def decode_rle(rle):
    if pd.isnull(rle):
        return

    return rle_decode(rle)

# Function to create mask image
def create_mask_image(mask, width, height):
    """
    Create a mask image from a run-length encoded (RLE) format.
    
    Args:
    - mask (str): The RLE-encoded mask string.
    - width (int): Width of the mask image.
    - height (int): Height of the mask image.
    
    Returns:
    - mask_image (numpy.ndarray or None): The mask image as a numpy array, or None if the decoded mask is NaN.
    """
    decoded_mask = decode_rle(mask)
    if pd.isnull(decoded_mask):
        return
    mask_image = np.zeros((height, width))
    mask_image[np.array(decoded_mask[::2]) - 1, np.array(decoded_mask[1::2]) - 1] = 1
    return mask_image

def load_model(path, pretrained=True):
    """
    Load a model from a given path.
    
    Args:
    - path (str): The path to the model file.
    - pretrained (bool): Whether to load pretrained weights.
    
    Returns:
    - model: The loaded model.
    """
    if pretrained:
        model = build_model()
        model.load_state_dict(torch.load(path))
        model.eval()
    else:
        model = torch.load(path)
    return model

def build_model():
    """
    Build a U-Net model.
    
    Returns:
    - model: The U-Net model.
    """
    model = smp.Unet(
        encoder_name=CFG.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(CFG.device)
    return model

import io
import base64
def show_img_v2(img, mask=None):
    """
    Display an image with an optional mask overlay and return the image encoded as base64.
    
    Args:
    - img (numpy.ndarray): The input image.
    - mask (numpy.ndarray or None): The mask image to overlay on the input image, or None if no mask is provided.
    
    Returns:
    - str: Base64-encoded image data.
    """
    fig = plt.figure()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    buf = io.BytesIO() # in-memory files
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='bone')
    
    if mask is not None:
        ax.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
    ax.axis('off')
    plt.savefig(buf, format="png")
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("ascii") # encode to html elements
    buf.close()
    return "data:image/png;base64,{}".format(data)

def plot_single(img, pred):
    """
    Plot a single image and its corresponding prediction.
    
    Args:
    - img (torch.Tensor): The input image tensor.
    - pred (torch.Tensor): The predicted mask tensor.
    
    Returns:
    - str: Base64-encoded image data.
    """
    img = img.cpu().detach()
    pred = pred.cpu().detach()

    img = img[0,].permute((1,2,0)).numpy() * 255
    img = img.astype('uint8')
    msk = pred[0,].permute((1,2,0)).numpy() * 255

    out = show_img_v2(img, msk)
    return out


if  __name__ == "__main__":
    plot_singles()
