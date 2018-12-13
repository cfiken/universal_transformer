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
from multiprocessing import Process

import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
import numpy as np
import matplotlib as plt

from model.transformer import Transformer
from datasource.sample_ds import SampleDataSource

tf.enable_eager_execution()

hparams = AttrDict()
hparams.num_layers = 4
hparams.num_units = 512
hparams.num_filter_units = hparams.num_units * 4
hparams.num_heads = 8
hparams.dropout_rate = 0.1
hparams.max_length = 50
hparams.batch_size = 32
hparams.learning_rate = 0.001
hparams.warmup_steps = 4000
hparams.num_epochs = 50
hparams.vocab_size = 3278
hparams.data_path = './data/'
hparams.ckpt_path = './ckpt/vanilla/l{}_u{}/model.ckpt'.format(hparams.num_layers, hparams.num_units)
hparams.log_dir = './logs/vanilla/l{}_u{}'.format(hparams.num_layers, hparams.num_units)
hparams1 = hparams

hparams2 = AttrDict()
hparams2.num_layers = 6
hparams2.num_units = 512
hparams2.num_filter_units = hparams2.num_units * 4
hparams2.num_heads = 8
hparams2.dropout_rate = 0.1
hparams2.max_length = 50
hparams2.batch_size = 64
hparams2.learning_rate = 0.001
hparams2.warmup_steps = 4000
hparams2.num_epochs = 30
hparams2.vocab_size = 3278
hparams2.data_path = './data/'
hparams2.ckpt_path = './ckpt/vanilla/l{}_u{}/model.ckpt'.format(hparams2.num_layers, hparams2.num_units)
hparams2.log_dir = './logs/vanilla/l{}_u{}'.format(hparams2.num_layers, hparams2.num_units)

# eager
def worker(hparams, gpu_id):
    with tf.device('/gpu:{}'.format(gpu_id)):
        ds = SampleDataSource(hparams)
        model = Transformer(hparams, True)
        optimizer = tf.train.AdamOptimizer(model.learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-09)
        model.load(optimizer)
        writer = tf.contrib.summary.create_file_writer(hparams['log_dir'])
        writer.set_as_default()
        model.fit(ds, optimizer, writer)

# graph mode
def worker_graph(hparams, gpu_id):
    gpu_id = 1
    with tf.Graph().as_default():
        with tf.device('/gpu:{}'.format(gpu_id)):
            ds = SampleDataSource(hparams)
            model = Transformer(hparams, True)
            model.build_graph()
            learning_rate = model.learning_rate()
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-09)
            tf_config = tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=tf.GPUOptions(
                    allow_growth=True
                )
            )
            with tf.Session(config=tf_config) as sess:
                sess.run(tf.global_variables_initializer())
                for e in range(hparams['num_epochs']):
                    ds.shuffle()
                    batch = ds.feed_dict(model, hparams['batch_size'], True)
                    start = time.time()
                    for b in batch:
                        inputs, targets = b[0], b[2]
                        loss_op = model.loss_op
                        grads = tf.gradients(loss_op, tf.trainable_variables())
                        train_op = optimizer.apply_gradients(zip(grads, tf.trainable_variables()), model.global_step)

                        _, loss, acc = sess.run([train_op, model.loss_op, model.acc_op], feed_dict={
                            model.encoder_inputs_ph: inputs,
                            model.decoder_inputs_ph: targets,
                            model.is_training_ph: True
                        })
                        step = sess.run(model.global_step)
                        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
                            tf.contrib.summary.scalar('summary/acc', acc)
                            tf.contrib.summary.scalar('summary/loss', loss)
                            tf.contrib.summary.scalar('summary/learning_rate', model.learning_rate())
                    print('elapsed: ', time.time() - start)
                    model.save(optimizer)
                    print('{} epoch finished. now {} step, loss: {:.4f}, acc: {:.4f}'.format(e, step, loss ,acc))

process_0 = Process(target=worker,args=(hparams1, 1))
#process_1 = Process(target=worker,args=(hparams2, 1))

process_0.start()

process_1.start()




