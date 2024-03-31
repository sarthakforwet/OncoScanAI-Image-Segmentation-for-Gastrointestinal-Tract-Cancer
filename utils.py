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
    """
    Configuration class for model training parameters.
    """
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
    '''
    Returns the mask corresponding to the inputted segmentation.

    Parameters:
        segmentation (str): A list of start points and lengths.
        shape (tuple): The shape to be taken by the mask.

    Returns:
        np.array: A 2D mask.
    '''
    # Get a list of numbers from the initial segmentation
    segm = np.asarray(segmentation.split(), dtype=int)

    # Get start point and length between points
    start_point = segm[0::2] - 1
    length_point = segm[1::2]

    # Compute the location of each endpoint
    end_point = start_point + length_point

    # Create an empty list mask the size of the original image
    # take only
    case_mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    # Change pixels from 0 to 1 that are within the segmentation
    for start, end in zip(start_point, end_point):
        case_mask[start:end] = 1

    case_mask = case_mask.reshape((shape[0], shape[1]))
    
    return case_mask

def get_id_mask(ID, train, verbose=False):
    '''
    Returns a mask for each case ID. If no segmentation was found, the mask will be empty
    - meaning formed by only 0.

    Parameters:
        ID (int): The case ID from the train.csv file.
        train (pd.DataFrame): DataFrame containing training data.
        verbose (bool): True if we want any prints.

    Returns:
        np.array: Segmentation mask.
    '''
    # Get the portion of dataframe where we have ONLY the specified ID
    ID_data = train[train["id"]==ID].reset_index(drop=True)

    # Split the dataframe into 3 series of observations
    # each for one specific class - "large_bowel", "small_bowel", "stomach"
    observations = [ID_data.loc[k, :] for k in range(3)]

    # Create the mask
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
    Function to Perform Run Length Decoding.
    
    Parameters:
        mask_rle (str): Run-length as string formatted (start length).
        shape (tuple): (height,width) of array to return.

    Returns:
        numpy array: 1 - mask, 0 - background.
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:-1][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


def decode_rle(rle):
    '''
    Function to decode run-length encoding.

    Parameters:
        rle (str): Run-length encoding.

    Returns:
        numpy array: Decoded mask image.
    '''
    if pd.isnull(rle):
        return

    return rle_decode(rle)

def create_mask_image(mask, width, height):
    '''
    Function to create mask image.

    Parameters:
        mask (str): Mask as run-length encoded string.
        width (int): Width of the mask image.
        height (int): Height of the mask image.

    Returns:
        numpy array: Mask image.
    '''
    decoded_mask = decode_rle(mask)
    if pd.isnull(decoded_mask):
        return
    mask_image = np.zeros((height, width))
    mask_image[np.array(decoded_mask[::2]) - 1, np.array(decoded_mask[1::2]) - 1] = 1
    return mask_image

def load_model(path, pretrained = True):
    '''
    Function to load a PyTorch model.

    Parameters:
        path (str): Path to the model file.
        pretrained (bool): Whether to load pre-trained weights.

    Returns:
        nn.Module: PyTorch model.
    '''
    if pretrained:
        model = build_model()
        model.load_state_dict(torch.load(path))
        model.eval()
    else:
        model = torch.load(path)
    return model

def build_model():
    '''
    Function to build a segmentation model.

    Returns:
        nn.Module: Segmentation model.
    '''
    model = smp.Unet(
        encoder_name=CFG.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(CFG.device)
    return model

def show_img_v2(img, mask=None):
    '''
    Function to display image with mask overlay.

    Parameters:
        img (numpy array): Input image.
        mask (numpy array): Mask image.

    Returns:
        str: Base64-encoded image for displaying in HTML.
    '''
    fig = plt.figure()
    buf = io.BytesIO() # in-memory files
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='bone')
    
    if mask is not None:
        ax.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
    ax.axis('off')
    plt.savefig(buf, format = "png")

    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    buf.close()
    return "data:image/png;base64,{}".format(data)

def plot_single(img, pred):
    '''
    Function to plot image with predicted mask.

    Parameters:
        img (torch.Tensor): Input image tensor.
        pred (torch.Tensor): Predicted mask tensor.

    Returns:
        str: Base64-encoded image for displaying in HTML.
    '''
    img = img.cpu().detach()
    pred = pred.cpu().detach()

    img = img[0,].permute((1,2,0)).numpy() * 255
    img = img.astype('uint8')
    msk = pred[0,].permute((1,2,0)).numpy() * 255

    out = show_img_v2(img, msk)
    return out
