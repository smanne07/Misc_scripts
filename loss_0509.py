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


def kl_loss_func():

    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)

    return kl_loss


def nb_loss_func(y_true, y_pred):
    
    log_mu = mu_hat_sf
    log_theta = theta_hat
    mu = mu_hat_exp_sf
    theta = theta_hat_exp
    f0 = - tf.math.lgamma(y_true + 1)
    f1 = - tf.math.lgamma(theta)
    f2 = tf.math.lgamma(y_true + theta)
    f3 = - (y_true + theta) * tf.math.log(theta + mu)
    f4 = theta * log_theta
    f5 = y_true * log_mu
    final = - K.sum(f0 + f1 + f2 + f3 + f4 + f5, axis = 1)

    return final



def nb_loss0_func(y_true, y_pred):
    """ Calculate Negative Binomial likelihood when reconstructing y with only b. """
    log_mu = mu_hat_mean_sf
    log_theta = theta_hat_mean
    mu = mu_hat_mean_exp_sf
    theta = theta_hat_mean_exp
    f0 = -1 * tf.math.lgamma(y_true + 1)
    f1 = -1 * tf.math.lgamma(theta)
    f2 = tf.math.lgamma(y_true + theta)
    f3 = - (y_true + theta) * tf.math.log(theta + mu)
    f4 = theta * log_theta
    f5 = y_true * log_mu
    final = - K.sum(f0 + f1 + f2 + f3 + f4 + f5, axis = 1)
    return final


