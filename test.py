# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sys
import os
import time
import random
import re
import json
import pickle
from typing import List, Tuple, Dict, Callable, Optional, Any, Sequence, Mapping, NamedTuple
from attrdict import AttrDict

import tensorflow as tf
import numpy as np
import matplotlib as plt

from model.attention import SelfAttention, MultiheadAttention
from model.embedding import EmbeddingSharedWeights
from model.ffn import FeedForwardNetwork
from model.layer_utils import LayerWrapper, LayerNormalization
from model import model_utils
from datasource.sample_ds import SampleDataSource

tf.enable_eager_execution()

hparams = AttrDict()
hparams.num_layers = 6
hparams.num_units = 256
hparams.num_filter_units = 1024
hparams.num_heads = 8
hparams.dropout_rate = 0.1
hparams.max_length = 50
hparams.batch_size = 64
hparams.vocab_size = 3278
hparams.data_path = './data'

ds = SampleDataSource(hparams)

class Transformer(tf.keras.Model):
    
    def __init__(self, hparams, is_train):
        super().__init__()
        self.hparams = hparams
        self.is_train = is_train
        self.embedding_layer = EmbeddingSharedWeights(hparams['vocab_size'], hparams['num_units'])
        self.encoder_stack = EncoderStack(hparams, is_train)
        self.encoder_embedding_dropout = tf.keras.layers.Dropout(hparams['dropout_rate'])
        
        self.decoder_stack = DecoderStack(hparams, is_train)
        self.decoder_embedding_dropout = tf.keras.layers.Dropout(hparams['dropout_rate'])
    
    def call(self, inputs, targets: Optional[np.ndarray] = None):
        attention_bias = model_utils.get_padding_bias(inputs)
        encoder_outputs = self._encode(inputs, attention_bias)
        
        if targets is None:
            return self.predict(encoder_outputs, attention_bias)
        else:
            logits = self._decode(encoder_outputs, targets, attention_bias)
            return logits
    
    def loss(self, inputs, targets):
        pad = tf.to_float(tf.not_equal(targets, 0))
        onehot_targets = tf.one_hot(targets, self.hparams['vocab_size'])
        logits = self(inputs, targets)
        cross_ents = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_targets,
            logits=logits
        )
        loss = tf.reduce_sum(cross_ents * pad) / tf.reduce_sum(pad)
        return loss
    
    def grads(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss = self.loss(inputs, targets)
        return tape.gradient(loss, self.variables)
        
    def predict(self, encoder_outputs, bias):
        pass
        
    def _encode(self, inputs, attention_bias):
        embedded_inputs = self.embedding_layer(inputs)
        embedded_inputs += model_utils.get_position_encoding(hparams['max_length'], hparams['num_units'])
        inputs_padding = model_utils.get_padding(inputs)

        if self.is_train:
            encoder_inputs = self.encoder_embedding_dropout(embedded_inputs)
        return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)
    
    def _decode(self, encoder_outputs, targets, attention_bias):
        decoder_inputs = self.embedding_layer(targets)
        decoder_inputs = tf.pad(decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        # add positional encoding
        length = decoder_inputs.shape[1]
        decoder_inputs += model_utils.get_position_encoding(length, self.hparams['num_units'])
        
        if self.is_train:
            decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)

        decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(length)
        outputs = self.decoder_stack(decoder_inputs, encoder_outputs, decoder_self_attention_bias, attention_bias)
        logits = self.embedding_layer.linear(outputs)
        return logits
        

class EncoderStack(tf.keras.Model):
    
    def __init__(self, hparams, is_train):
        super(EncoderStack, self).__init__()
        self.my_layers = []
        
        for i in range(hparams['num_layers']):
            self_attention_layer = SelfAttention(hparams['num_units'], hparams['num_heads'], hparams['dropout_rate'], is_train)
            ffn_layer = FeedForwardNetwork(hparams['num_units'], hparams['num_filter_units'], hparams['dropout_rate'], is_train)
            self.my_layers.append([
                LayerWrapper(self_attention_layer, hparams['num_units'], hparams['dropout_rate'], is_train),
                LayerWrapper(ffn_layer, hparams['num_units'], hparams['dropout_rate'], is_train),
            ])
            
        self.output_norm = LayerNormalization(hparams['num_units'])
            
    
    def call(self, encoder_inputs, attention_bias, inputs_padding):
        for n, layer in enumerate(self.my_layers):
            self_attention_layer = layer[0]
            ffn_layer = layer[1]
            
            encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
            encoder_inputs = ffn_layer(encoder_inputs)
            
        return self.output_norm(encoder_inputs)

class DecoderStack(tf.keras.Model):
    
    def __init__(self, hparams, is_train):
        super(DecoderStack, self).__init__()
        self.my_layers = []
        
        for i in range(hparams['num_layers']):
            self_attention_layer = SelfAttention(hparams['num_units'], hparams['num_heads'], hparams['dropout_rate'], is_train)
            enc_dec_attention_layer = MultiheadAttention(hparams['num_units'], hparams['num_heads'], hparams['dropout_rate'], is_train)
            ffn_layer = FeedForwardNetwork(hparams['num_units'], hparams['num_filter_units'], hparams['dropout_rate'], is_train)
            self.my_layers.append([
                LayerWrapper(self_attention_layer, hparams['num_units'], hparams['dropout_rate'], is_train),
                LayerWrapper(enc_dec_attention_layer, hparams['num_units'], hparams['dropout_rate'], is_train),
                LayerWrapper(ffn_layer, hparams['num_units'], hparams['dropout_rate'], is_train),
            ])
            
        self.output_norm = LayerNormalization(hparams['num_units'])
            
    
    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias, attention_bias):
        for n, layer in enumerate(self.my_layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            ffn_layer = layer[2]
            
            decoder_inputs = self_attention_layer(decoder_inputs, decoder_self_attention_bias)
            decoder_inputs = enc_dec_attention_layer(decoder_inputs, encoder_outputs, attention_bias)
            decoder_inputs = ffn_layer(decoder_inputs)
            
        return self.output_norm(decoder_inputs)

batch = ds.feed_dict(None, hparams['batch_size'], True)

one_batch = batch[0]

one_batch[2].shape

model = Transformer(hparams, True)

yay = model.grads(one_batch[0], one_batch[2])
yay

yay.numpy().shape


