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

tf.enable_eager_execution()

hparams = AttrDict()
hparams.num_layers = 6
hparams.num_units = 256
hparams.num_filter_units = 1024
hparams.num_heads = 8
hparams.dropout_rate = 0.1

class Transformer(tf.keras.Model):
    
    def __init__(self, hparams, is_train):
        self.is_train = is_train
    
    def call(self, inputs, targets: Optional[np.ndarray] = None):
        logits = inputs
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

test_text = [[1,2,5,8,9,12,5,1,0,0,0,0,0,0,0,0,0]]
attention_bias = model_utils.get_padding(test_text)

encoder = EncoderStack(hparams, True)

encoder(test_text, attention_bias, None)








