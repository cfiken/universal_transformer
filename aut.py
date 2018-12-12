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
hparams.num_units = 2048
hparams.num_filter_units = hparams.num_units * 4
hparams.num_heads = 8
hparams.dropout_rate = 0.1
hparams.max_length = 50
hparams.batch_size = 64
hparams.warmup_steps = 4000
hparams.num_epochs = 20
hparams.vocab_size = 3278
hparams.act_max_step = 20
hparams.act_epsilon = 0.01
hparams.act_loss_weight = 0.01
hparams.data_path = './data/'
hparams.ckpt_path = './ckpt/aut/u{}_actl{}/model.ckpt'.format(hparams.num_units, hparams.act_loss_weight)
hparams.log_dir = './logs/aut/u{}_actl{}'.format(hparams.num_units, hparams.act_loss_weight)
hparams1 = hparams

hparams2 = AttrDict()
hparams2.num_units = 2048
hparams2.num_filter_units = hparams2.num_units * 4
hparams2.num_heads = 8
hparams2.dropout_rate = 0.1
hparams2.max_length = 50
hparams2.batch_size = 64
hparams2.warmup_steps = 4000
hparams2.num_epochs = 20
hparams2.vocab_size = 3278
hparams.act_max_step = 20
hparams.act_epsilon = 0.01
hparams.act_loss_weight = 0.001
hparams2.data_path = './data/'
hparams2.ckpt_path = './ckpt/aut/u{}_actl{}/model.ckpt'.format(hparams2.num_units, hparams.act_loss_weight)
hparams2.log_dir = './logs/aut/u{}_actl{}'.format(hparams2.num_units, hparams.act_loss_weight)
hparams2 = hparams2

# eager
def worker(hparams, gpu_id):
    with tf.device('/gpu:{}'.format(gpu_id)):
        ds = SampleDataSource(hparams)
        model = UniversalTransformer(hparams, True)
        optimizer = tf.train.AdamOptimizer(model.learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-09)
        model.load(optimizer)
        writer = tf.contrib.summary.create_file_writer(hparams['log_dir'])
        writer.set_as_default()
        model.fit(ds, optimizer, writer)

worker(hparams1, 2)


