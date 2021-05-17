# -*- coding: utf-8 -*-
"""
Created on Tue Aug  11 11:20:03 2020

@author: jane
"""

import numpy as np
import anndata as ann
import random
import os
import itertools
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


def preprocess_data(adata, filter_gene = False, filter_cell = False, 
                    gene_counts = 5, cell_counts = 1, copy_adata = True,
                    counts_per_cell_after = 10000):

    if filter_gene:
        sc.pp.filter_genes(adata, min_counts = gene_counts)

    if filter_cell:
        sc.pp.filter_cells(adata, min_counts = cell_counts)
    
    if copy_adata:
        adata.raw = adata.copy()
    
    n_counts = adata.X.sum(axis = 1)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after = counts_per_cell_after)
    adata.obs['size_factors'] = n_counts / counts_per_cell_after
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    return adata


def calc_B(adata, condition_key = 'batch'):

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(adata.obs[condition_key])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    n_counts = adata.obs['n_counts'].to_numpy()
    loglib = np.log10(n_counts)
    loglib = loglib.reshape((len(loglib), 1))

    #B_raw = np.concatenate(( onehot_encoded, loglib), axis=1)
    B_raw = onehot_encoded
    B = scale(B_raw)
    adata.obsm['B_raw'] = B_raw
    adata.obsm['B'] = B
    adata.obsm['loglib'] = loglib

    return adata



def split_data(adata, nfolds = 6):

    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)

    test_frac = 1 / nfolds
    test_size = int(adata.shape[0] * test_frac)

    test_idx_list = []
    for i in np.arange(nfolds - 1):
        start_id = i * test_size
        end_id = (i + 1) * test_size 
        test_idx = indices[ start_id : end_id ]
        test_idx_list.append(test_idx)

    start_id = (nfolds - 1) * test_size
    test_idx_list.append(indices[start_id:])

    return test_idx_list



def train_test_split(adata, test_idx):

    train_idx = np.setdiff1d(np.arange(adata.shape[0]), test_idx)
    train_data = adata[train_idx, :]
    valid_data = adata[test_idx, :]

    return train_data, valid_data




def prep_in_out(adata, condition_key = 'batch'):

    z_blank = np.zeros((adata.n_obs, 30), dtype=np.float32)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(adata.obs[condition_key])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    n_counts = adata.obs['n_counts'].to_numpy()
    loglib = np.log10(n_counts)
    loglib = loglib.reshape((len(loglib), 1))

    B_raw = np.concatenate(( onehot_encoded, loglib), axis=1)
    B = scale(B_raw)

    inputs = [z_blank, adata.X, B, np.log(adata.obs.size_factors), B, adata.raw.X]
    outputs = [adata.raw.X, adata.raw.X]

    return inputs, outputs




def prep_in_out_v2(adata, reference_batch_num = None, B_raw_key = 'B_raw', loglib_key = 'loglib'):

    z_blank = np.zeros((adata.n_obs, 30), dtype=np.float32)


    if reference_batch_num is not None:
        reference_like_zero = np.zeros((adata.obsm[B_raw_key].shape[0], reference_batch_num), dtype=np.float32)
        B_targettrain = np.concatenate(( adata.obsm[loglib_key], reference_like_zero, adata.obsm[B_raw_key]), axis=1)
    else:
        B_targettrain = np.concatenate(( adata.obsm[loglib_key], adata.obsm[B_raw_key]), axis=1)

    
    inputs = [z_blank, adata.X, B_targettrain, np.log(adata.obs.size_factors), B_targettrain, adata.raw.X]
    outputs = [adata.raw.X, adata.raw.X]

    return inputs, outputs


def calc_nb_loss(y, mu, theta):
    """ Calculate Negative Binomial likelihood """
    y = tf.cast(y, np.float32)
    #mu = np.where(mu != 0, mu, 1e-7)
    mu = tf.where(tf.equal(mu, 0.), 1e-7, mu)
    theta = tf.where(tf.equal(theta, 0.), 1e-7, theta)
    #theta = np.where(theta != 0, theta, 1e-7)
    log_mu = tf.math.log(mu)
    log_theta = tf.math.log(theta)
    f0 = - tf.math.lgamma(y + 1)
    f1 = - tf.math.lgamma(theta)
    f2 = tf.math.lgamma(y + theta)
    f3 = - (y + theta) * tf.math.log(theta + mu)
    f4 = theta * log_theta
    f5 = y * log_mu
    final = - (f0 + f1 + f2 + f3 + f4 + f5)
    final = final.numpy()
    return final


def calc_label_out(adata, condition_key = 'major_cell_type'):

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(adata.obs[condition_key])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded 


def savercat_preprocess(adata, predict_key, adjust_key):

    adata.obsm['saver_targetL'] = calc_label_out(adata, condition_key=predict_key)
    adata = preprocess_data(adata)
    adata = calc_B(adata, condition_key = adjust_key)
    B = np.concatenate(( adata.obsm['loglib'], adata.obsm['B_raw']), axis=1)
    adata.obsm['saver_batch'] = B

    return adata






