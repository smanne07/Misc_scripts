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
from utils_0509 import *
from layers_0509 import *



""" Network. """

class CVAE():

    def __init__(self, 
                 x_input_size,
                 b_input_size,
                 lb_input_size,
                 sf_input_size = 1,
                 enc = (256, 256, 128),
                 dec = (128, 256, 256),
                 latent_k = 30,
                 alpha = 0.01,
                 input_dropout = 0.,
                 encoder_dropout = 0.1,
                 nonmissing_indicator = None,
                 init = tf.keras.initializers.Orthogonal(),
                 optimizer = None,
                 lr = 0.001,
                 clipvalue = 5,
                 clipnorm = 1,
                 theta_min = 1e-6,
                 theta_max = 1e2):

        self.x_input_size = x_input_size
        self.b_input_size = b_input_size
        self.lb_input_size = lb_input_size
        self.z_input_size = latent_k
        self.sf_input_size = sf_input_size
        self.disp_input_size = b_input_size
        self.enc = enc
        self.dec = dec
        self.latent_k = latent_k
        self.alpha = alpha
        self.input_dropout = input_dropout
        self.encoder_dropout = encoder_dropout
        self.init = init
        self.lr = lr
        self.clipvalue = clipvalue
        self.clipnorm = clipnorm
        self.theta_min = theta_min
        self.theta_max = theta_max



        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr, 
                                                  clipnorm = clipnorm, clipvalue = clipvalue)
        else:
            self.optimizer = optimizer

        
        self.extra_models = {}
        self.model = None


    
    def build(self, print_model = False):


        """ Inputs. """
        self.x_input = Input(shape = (self.x_input_size, ), name = 'x_input')
        self.b_input = Input(shape = (self.b_input_size, ), name = 'B')
        self.sf_input = Input(shape = (self.sf_input_size, ), name = 'sf_input')
        self.z_input = Input(shape = (self.z_input_size, ), name = 'z_input')
        self.disp_input = Input(shape = (self.disp_input_size, ), name = 'nb_input')
        self.x_raw_input = Input(shape = (self.x_input_size, ), name = 'x_raw_input')
        self.lb_input = Input(shape = (self.lb_input_size, ), name = 'lb_input')



        """ Build the encoder. """
        self.z = keras.layers.concatenate([self.x_input, self.b_input])

        for i, hid_size in enumerate(self.enc):
            dense_layer_name = 'e%s' % i
            bn_layer_name = 'be%s' % i
            self.z = Dense(hid_size, activation = None, use_bias = True, 
                        kernel_initializer = self.init, name = dense_layer_name)(self.z)
            self.z = LeakyReLU(alpha = 0.01)(self.z)
            self.z = BatchNormalization(center = False, scale = True, name = bn_layer_name)(self.z)
            if i == 0:
                self.z = Dropout(self.encoder_dropout)(self.z)
            
        self.z_mean = Dense(self.latent_k, activation = None, use_bias = True, 
                            kernel_initializer = self.init, name = 'z_mean_dense')(self.z)
        self.z_mean = LeakyReLU(alpha = 0.01, name = 'z_mean_act')(self.z_mean)
        self.z_mean = BatchNormalization(center = False, scale = True, name = 'bz')(self.z_mean)
        self.z_log_var = Dense(self.latent_k, activation = None, use_bias = True, 
                            kernel_initializer = tf.keras.initializers.Orthogonal(gain = 0.01), 
                            name = 'z_log_var')(self.z)

        # Sampling latent space
        self.z_out = Lambda(sample_z, output_shape = (self.latent_k, ))([self.z_mean, self.z_log_var])

        self.extra_models['mean_out'] = Model([self.x_input, self.b_input], self.z_mean, name = 'mean_out')
        self.extra_models['var_out'] = Model([self.x_input, self.b_input], self.z_log_var, name = 'var_out')
        self.extra_models['samp_out'] = Model([self.x_input, self.b_input], self.z_out, name = 'samp_out')


        """ Build the prediction network. """
        self.lb_pred = Dense(self.latent_k, activation = 'sigmoid', use_bias = True, 
                            kernel_initializer = self.init, name = 'pred_sigmoid')(self.z_mean)
        self.lb_pred = BatchNormalization(center = False, scale = True, name = 'lz1')(self.lb_pred)
        self.lb_pred = Dense(int(0.5*self.latent_k), activation = 'sigmoid', use_bias = True, 
                            kernel_initializer = self.init, name = 'pred_sigmoid2')(self.lb_pred)
        self.lb_pred = BatchNormalization(center = False, scale = True, name = 'lz2')(self.lb_pred)
        self.lb_pred = Dense(self.lb_input_size, activation = 'softmax', use_bias = True, 
                            kernel_initializer = self.init, name = 'pred_softmax')(self.lb_pred)
        self.extra_models['lb_pred'] = Model([self.x_input, self.b_input], self.lb_pred, name = 'lb_pred')


        """ Build the decoder. """
        #### decoder network
        self.decoder_dense_layers = []
        self.decoder_leaky_layers = []
        for i, hid_size in enumerate(self.dec):
            dense_layer_name = 'd%s' % i
            self.decoder_dense_layers.append ( Dense(hid_size, activation = None, use_bias = True, 
                                                kernel_initializer = self.init, name = dense_layer_name) )
            self.decoder_leaky_layers.append ( LeakyReLU(alpha = 0.01) )
        self.last_layer_mu = Dense(self.x_input_size, activation = None, use_bias = True, 
                                kernel_initializer = self.init, name = 'mu_out')


        #### start from sampled latent values
        self.decoder11 = keras.layers.concatenate([self.z_out, self.b_input])
        for i, hid_size in enumerate(self.dec):
            self.decoder11 = self.decoder_dense_layers[i](self.decoder11)
            self.decoder11 = self.decoder_leaky_layers[i](self.decoder11)
        self.mu_hat = self.last_layer_mu(self.decoder11)
        self.mu_hat_sf = AddLayer(name = 'mu_hat_sf')([self.mu_hat, self.sf_input])
        self.mu_hat_exp_sf = ExpLayer(name = 'mu_hat_exp_sf')(self.mu_hat_sf)
        self.mu_hat_exp = ExpLayer(name = 'mu_hat_exp')(self.mu_hat)


        #### start from zeroed latent values
        self.decoder12_mean = keras.layers.concatenate([self.z_input, self.b_input])
        for i, hid_size in enumerate(self.dec):
            self.decoder12_mean = self.decoder_dense_layers[i](self.decoder12_mean)
            self.decoder12_mean = self.decoder_leaky_layers[i](self.decoder12_mean)
        self.mu_hat_mean = self.last_layer_mu(self.decoder12_mean)
        self.mu_hat_mean_sf = AddLayer(name = 'mu_hat_mean_sf')([self.mu_hat_mean, self.sf_input])
        self.mu_hat_mean_exp_sf = ExpLayer(name = 'mu_hat_mean_exp_sf')(self.mu_hat_mean_sf)
        self.mu_hat_mean_exp = ExpLayer(name = 'mu_hat_mean_exp')(self.mu_hat_mean)

        self.extra_models['decoder_mean'] = Model([self.z_input, self.b_input], [self.mu_hat_mean_exp], name = 'decoder_mean')



        """ Build the dispersion network. """
        self.last_layer_theta = Dense(self.x_input_size, activation = None, use_bias = True, 
                                      kernel_initializer = self.init, name = 'theta_out')

        #### start from sampled latent values
        self.theta_hat = self.last_layer_theta(self.disp_input)
        self.theta_hat = ClipLayer(name = 'clip_theta_hat')(self.theta_hat)
        self.theta_hat_exp = ExpLayer(name = 'theta_hat_exp')(self.theta_hat)

        #### start from zeroed latent values
        self.theta_hat_mean = self.last_layer_theta(self.disp_input)
        self.theta_hat_mean = ClipLayer(name = 'clip_theta_hat_mean')(self.theta_hat_mean)
        self.theta_hat_mean_exp = ExpLayer(name = 'theta_hat_mean_exp')(self.theta_hat_mean)

        self.extra_models['disp_model'] = Model(self.disp_input, self.theta_hat_mean_exp, name = 'disp_model')



        """ Build the whole network. """
        # decoder output
        self.out_hat = keras.layers.concatenate([self.mu_hat_sf, self.theta_hat], name = 'out')
        self.out_hat_mean = keras.layers.concatenate([self.mu_hat_mean_sf, self.theta_hat_mean], name = 'out_mean')
        # the whole model
        self.model = Model(inputs = [self.z_input, self.x_input, self.b_input, self.sf_input, self.disp_input, self.x_raw_input, self.lb_input], 
                           outputs = [self.out_hat, self.out_hat_mean, self.lb_pred], 
                           name = 'model')

        if print_model:
            self.model.summary()


        self.pred_loss = K.sum( tf.keras.losses.categorical_crossentropy(self.lb_input, self.lb_pred), axis = -1)
        self.kl_loss = -0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis = -1)
        self.recon_loss = ((1 - self.alpha) * self.nb_loss_func(self.x_raw_input, self.mu_hat_exp_sf) 
                            + self.alpha * self.nb_loss0_func(self.x_raw_input, self.mu_hat_mean_exp_sf))
        


    def add_loss(self, pred_weight, kl_weight=1):
        self.final_loss = kl_weight * self.kl_loss + self.recon_loss + pred_weight * self.pred_loss
        self.model.add_loss(self.final_loss)
        self.model.add_metric(self.pred_loss, name='pred_loss')
        self.model.add_metric(self.kl_loss, name='kl_loss')
        self.model.add_metric(self.recon_loss, name='recon_loss')



    def compile_model(self, pred_weight, kl_weight=1, optimizer = None):

        self.add_loss(pred_weight, kl_weight)

        if optimizer is not None:
            self.optimizer = optimizer

        self.model.compile(optimizer = self.optimizer)



    def kl_loss_func(self):

        kl_loss = -0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis = -1)

        return kl_loss



    def nb_loss_func(self, y_true, y_pred):
    
        log_mu = self.mu_hat_sf
        log_theta = self.theta_hat
        mu = self.mu_hat_exp_sf
        theta = self.theta_hat_exp
        f0 = -1 * tf.math.lgamma(y_true + 1)
        f1 = -1 * tf.math.lgamma(theta)
        f2 = tf.math.lgamma(y_true + theta)
        f3 = - (y_true + theta) * tf.math.log(theta + mu)
        f4 = theta * log_theta
        f5 = y_true * log_mu
        final = - K.sum(f0 + f1 + f2 + f3 + f4 + f5, axis = 1)

        return final



    def nb_loss0_func(self, y_true, y_pred):
        
        log_mu = self.mu_hat_mean_sf
        log_theta = self.theta_hat_mean
        mu = self.mu_hat_mean_exp_sf
        theta = self.theta_hat_mean_exp
        f0 = -1 * tf.math.lgamma(y_true + 1)
        f1 = -1 * tf.math.lgamma(theta)
        f2 = tf.math.lgamma(y_true + theta)
        f3 = - (y_true + theta) * tf.math.log(theta + mu)
        f4 = theta * log_theta
        f5 = y_true * log_mu
        final = - K.sum(f0 + f1 + f2 + f3 + f4 + f5, axis = 1)

        return final


    def load_weights(self, filename):

        self.model.load_weights(filename)


    def save_weights(self, filename, save_extra = False, extra_filenames = None):

        self.model.save_weights(filename)

        if save_extra:
            self.extra_models['mean_out'].save_weights(extra_filenames["mean_out"])
            self.extra_models['var_out'].save_weights(extra_filenames["var_out"])
            self.extra_models['samp_out'].save_weights(extra_filenames["samp_out"])
            self.extra_models['disp_model'].save_weights(extra_filenames["disp_model"])
            self.extra_models['decoder_mean'].save_weights(extra_filenames["decoder_mean"])


    def predict_latent(self, X, B):

        latent_mean = self.extra_models['mean_out'].predict([X, B])

        return latent_mean


    
    def predict_beta(self, X, B, sf):

        zmean = self.extra_models['mean_out'].predict([X, B])
        X_lambda = self.extra_models['decoder_mean'].predict([zmean, B])
        X_theta = self.extra_models['disp_model'].predict(B)
        X_lambda = (X_lambda.T * sf).T

        return X_lambda, X_theta



    def model_initialize(self, adata, 
                         epochs=300, batch_size=64, 
                         validation_split=0.1, 
                         shuffle=True, fit_verbose=1, 
                         lr_patience=1, 
                         lr_factor=0.1, 
                         lr_verbose=True,
                         es_patience=2,
                         es_verbose=True):

        callbacks = []
        lr_cb = ReduceLROnPlateau(monitor='val_pred_loss', patience=lr_patience, factor=lr_factor, verbose=lr_verbose)
        callbacks.append(lr_cb)
        es_cb = EarlyStopping(monitor='val_pred_loss', patience=es_patience, verbose=es_verbose)
        callbacks.append(es_cb)

        z_blank = np.zeros((adata.n_obs, self.latent_k), dtype=np.float32)
        inputs = [z_blank, 
                  adata.X, 
                  adata.obsm['saver_batch'], 
                  np.log(adata.obs.size_factors), 
                  adata.obsm['saver_batch'], 
                  adata.raw.X, 
                  adata.obsm['saver_targetL']]
        outputs = [adata.raw.X, 
                  adata.raw.X, 
                  adata.obsm['saver_targetL']]

        loss = self.model.fit(inputs, outputs,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              callbacks=callbacks,
                              validation_split=validation_split,
                              verbose=fit_verbose)


        return loss




    def model_finetune(self, adata, 
                         epochs=300, batch_size=64, 
                         validation_split=0.1, 
                         shuffle=True, fit_verbose=1, 
                         lr_patience=4, 
                         lr_factor=0.1, 
                         lr_verbose=True,
                         es_patience=6,
                         es_verbose=True):

        callbacks = []
        lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=lr_patience, factor=lr_factor, verbose=lr_verbose)
        callbacks.append(lr_cb)
        es_cb = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=es_verbose)
        callbacks.append(es_cb)

        z_blank = np.zeros((adata.n_obs, self.latent_k), dtype=np.float32)
        inputs = [z_blank, 
                  adata.X, 
                  adata.obsm['saver_batch'], 
                  np.log(adata.obs.size_factors), 
                  adata.obsm['saver_batch'], 
                  adata.raw.X, 
                  adata.obsm['saver_targetL']]
        outputs = [adata.raw.X, 
                  adata.raw.X, 
                  adata.obsm['saver_targetL']]

        loss = self.model.fit(inputs, outputs,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              callbacks=callbacks,
                              validation_split=validation_split,
                              verbose=fit_verbose)

        return loss







  