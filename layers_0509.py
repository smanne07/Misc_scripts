# -*- coding: utf-8 -*-
"""
Created on Tue Aug  11 11:20:03 2020

@author: jane
"""

import random
import os
import numpy as np
import scanpy as sc
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.utils.vis_utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, LeakyReLU, Lambda
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as pl


def sample_z(args):
    mu, log_sigma = args
    batch = tf.shape(mu)[0]
    dim = tf.shape(mu)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * epsilon

def addsf(args):
    logmu, logsf = args
    final = tf.math.add(logmu, logsf)
    return final

ClipLayer = lambda name: Lambda(lambda x: tf.clip_by_value(x, 1e-6, 1e2), name = name)
ExpLayer = lambda name: Lambda(lambda x: tf.math.exp(x), name = name)


AddLayer = lambda name: Lambda(lambda x: addsf(x), name = name)




