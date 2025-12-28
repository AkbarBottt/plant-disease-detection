# ===============================
# Import Necessaries
# ===============================

# Basic utilities
import os
import itertools
import warnings
warnings.filterwarnings("ignore")

# Data handling
import numpy as np
import pandas as pd

# Image processing
import cv2
from tensorflow.keras.preprocessing import image

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import missingno as msno
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Deep Learning (TensorFlow & Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, Activation
)
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Transfer Learning - EfficientNet
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2,
    EfficientNetB3, EfficientNetB4, EfficientNetB5,
    EfficientNetB6, EfficientNetB7
)

# ===============================
# Environment Info
# ===============================
print("Python version      :", os.sys.version)
print("TensorFlow version  :", tf.__version__)
print("Modules loaded successfully")

