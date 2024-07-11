import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch.nn.functional as F
import math
import numpy as np
import random
from torch import nn
from torch.optim import Adam
import streamlit as st
from PIL import Image, ImageOps
import os
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from torch.nn.modules.linear import Linear


BATCH_SIZE = 64
img_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
