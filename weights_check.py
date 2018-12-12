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

import tensorflow as tf
import numpy as np
from attrdict import AttrDict

from model.ut import UniversalTransformer
from datasource.sample_ds import SampleDataSource

tf.enable_eager_execution()

hparams = AttrDict()
#hparams.num_layers = 4
hparams.num_units = 1536
hparams.num_filter_units = hparams.num_units * 4
hparams.num_heads = 8
hparams.dropout_rate = 0.1
hparams.max_length = 50
hparams.batch_size = 2
hparams.learning_rate = 0.001
hparams.warmup_steps = 4000
hparams.num_epochs = 2
hparams.vocab_size = 3278
hparams.data_path = './data/'
hparams.ckpt_path = './ckpt/aut/u{}/model.ckpt'.format(hparams.num_units)
hparams.log_dir = './logs/aut/u{}'.format(hparams.num_units)
hparams.act_max_step = 20
hparams.act_epsilon = 0.01
hparams.act_loss_weight = 0.01
hparams1 = hparams

# Universal Transformer
gpu_id = 1
with tf.device('/gpu:{}'.format(gpu_id)):
    ds = SampleDataSource(hparams)
    batch = ds.feed_dict(None, hparams['batch_size'], True)
    model = UniversalTransformer(hparams, True)
    a, b, c = batch[0]
    _ = model(a, c)
    weights = model.weights

len(weights)

total = 0
for w in weights:
    if len(w.shape) == 1:
        total += w.shape[0].value
    elif len(w.shape) == 2:
        total += w.shape[0].value * w.shape[1].value
print(total)


