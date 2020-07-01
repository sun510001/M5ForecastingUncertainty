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
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import itertools as it

print('TF version', tf.__version__)

# VER = 'v1.0'  # normal
# VER = 'v1.1'  # LAGS = [28, 35]
# VER = 'v1.2'  # LAGS add 7

# VER = 'v1.4'  # LAGS [28, 29, 30, 31, 32, 33]
# VER = 'v1.5'  # dropout = 0.2 * 3 0.2074
# VER = 'v1.6'  # dropout = 0.2 * 2 0.2058
# VER = 'v1.7'  # min_lr = 1e-3 dropout = 0.2 * 2 0.2058
# VER = 'v1.8'  # batch_size=10_000
# VER = 'v2.0'  # change loss and model for RNN val_loss: 0.207?
# VER = 'v2.1'  # not changing for RNN val_loss: 0.2034
# VER = 'v2.2'  # add features x_28_month_mean, x_28_month_max, x_28_month_min, x_28_month_max_to_min_diff vl: 0.1920
# VER = 'v2.3'  # load csv directly
# VER = 'v2.4'  # epoch 1000 vl: 0.1970 best!
# VER = 'v2.5'  # epoch 500 vl: 0.2037 -> epoch 1000
# VER = 'v3.0'    # feature 9 ls: 0.1890
# ["x_28_month_mean", "x_28_month_max", "x_28_month_min", "x_28_month_max_to_min_diff",
# 'x_28_wk_mean', 'x_28_wk_median', 'x_28_wk_max', 'x_28_wk_min',
# 'x_28_wk_max_to_min_diff']
# VER = '2.6'     # epoch 1000 f4 month loss: 0.1976
# VER = 'v3.1'  # f9 lr=1-e3 epoch=1000
# VER = 'v2.7'  # f4 lr=1-e4 epoch=500
# VER = 'v2.8'    # f4 lr=1-e4 epoch=30 BATCH_SIZE = 700
# VER = 'v3.2'  # f4+'x_28_wk_mean', 'x_28_wk_median' lr=5-e4 epoch=30 BATCH_SIZE = 700; until 20:32
# VER = 'v3.3'    # f9 EPOCH = 35 MIN_LR = 1e-4 BATCH_SIZE = 1000 loss: 0.1793
# VER = 'v3.4'    # f9 EPOCH = 50 MIN_LR = 1e-4 BATCH_SIZE = 10000 loss: 0.1793
# VER = 'v3.5'    # x = L.Bidirectional(L.GRU(128, return_sequences=True, name="d2"))(x)
# VER = 'v3.6'    # dropout=0.2 MIN_LR = 5e-4
# VER = 'v3.7'      # without features
# VER = 'v3.8'        # LSTM
# VER = 'v4.0'    # cnn normal EPOCH = 50 MIN_LR = 1e-4 BATCH_SIZE = 10_000
# VER = 'v4.1'    # BATCH_SIZE = 15_000 dropout 0.2 add one dense layer
# VER = 'v4.2'    # BATCH_SIZE = 10_000 dorpout 0.3 2 dense layer
# VER = 'v4.3'    # dropout0.25 dense 840 best 0.1987 pubLB:0.10096
# VER = 'v4.4'    # add one layer val_loss: 0.1963
# VER = 'v4.5'    # add plus 4 feature location*salesmeans; 4 layers
# VER = 'v4.6'    # best retrain val_loss: 0.1980
# VER = 'v4.7'  # 4 layers; EPOCH = 50; MIN_LR = 1e-3; BATCH_SIZE = 10_000; 9 features +2 var featrues
VER = 'v4.8'   # 4 layers; EPOCH = 50; MIN_LR = 1e-3; BATCH_SIZE = 10_000; 10 features var:0.1964 pubLB:0.10228

CATEGORIZE = True
START = 1400
UPPER = 1970
maps = {}
mods = {}
OUTPUT_MODEL = 'w_{0}.h5'.format(VER)
OUTPUT_IMAGE = 'image_{0}_{1}.png'
QUANTILES = ["0.005", "0.025", "0.165", "0.250", "0.500", "0.750", "0.835", "0.975", "0.995"]
VALID = []
EVAL = []

EPOCH = 50
MIN_LR = 1e-3
BATCH_SIZE = 10_000

LAGS = [28, 31, 35, 42, 49, 56, 63]
LEN = 1969 - START + 1
MAX_LAG = max(LAGS)
print(LEN, MAX_LAG)

NITEMS = 42840
CATCOLS = ['snap_CA', 'snap_TX', 'snap_WI', 'wday', 'month', 'year', 'event_name', 'nday',
           'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
LIST_OF_FEATURE = ["x_28_month_mean", "x_28_month_median", "x_28_month_max", "x_28_month_min",
                   "x_28_month_max_to_min_diff",
                   'x_28_wk_mean', 'x_28_wk_median', 'x_28_wk_min', 'x_28_wk_max', 'x_28_wk_max_to_min_diff']
# LIST_OF_FEATURE = ["x_28_month_mean", "x_28_month_max", "x_28_month_min", "x_28_month_max_to_min_diff",
#                    # "x_28_month_skew", "x_28_month_kurt",
#                    'x_28_wk_mean', 'x_28_wk_median', 'x_28_wk_max', 'x_28_wk_min', 'x_28_wk_max_to_min_diff']

# LIST_OF_FEATURE = ["x_28_month_mean", "x_28_month_max", "x_28_month_min", "x_28_month_max_to_min_diff"]
PROC_CSV_EXIST = True

ONLY_LOAD_MODEL = False
