# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import gc
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

VER = 'v1.0'
CATEGORIZE = True
START = 1400
UPPER = 1970
maps = {}
mods = {}
OUTPUT_MODEL = 'w_{0}.h5'.format(VER)
OUTPUT_IMAGE = 'image_{0}.png'.format(VER)
QUANTILES = ["0.005", "0.025", "0.165", "0.250", "0.500", "0.750", "0.835", "0.975", "0.995"]
VALID = []
EVAL = []