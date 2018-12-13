# -*- coding: utf-8 -*-
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
#hparams.num_layers = 4
hparams.num_units = 1024
hparams.num_filter_units = hparams.num_units * 4
hparams.num_heads = 8
hparams.dropout_rate = 0.1
hparams.max_length = 50
hparams.batch_size = 64
hparams.learning_rate = 0.001
hparams.warmup_steps = 4000
hparams.num_epochs = 2
hparams.vocab_size = 3278
hparams.data_path = './data/'
hparams.ckpt_path = './ckpt/aut/u{}_2/model.ckpt'.format(hparams.num_units)
hparams.log_dir = './logs/aut/u{}_2'.format(hparams.num_units)
hparams.act_max_step = 20
hparams.act_epsilon = 0.01
hparams.act_loss_weight = 0.01
hparams1 = hparams

hparams2 = AttrDict()
hparams2.num_layers = 6
hparams2.num_units = 1024
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

hparams3 = AttrDict()
hparams3.num_layers = 1
hparams3.num_units = 1024
hparams3.num_filter_units = hparams3.num_units * 4
hparams3.num_heads = 8
hparams3.dropout_rate = 0.1
hparams3.max_length = 50
hparams3.batch_size = 64
hparams3.learning_rate = 0.001
hparams3.warmup_steps = 4000
hparams3.num_epochs = 20
hparams3.vocab_size = 3278
hparams3.data_path = './data/'
hparams3.ckpt_path = './ckpt/vanilla/l{}_u{}/model.ckpt'.format(hparams3.num_layers, hparams3.num_units)
hparams3.log_dir = './logs/vanilla/l{}_u{}'.format(hparams3.num_layers, hparams3.num_units)

ds = SampleDataSource(hparams)

class UniversalTransformer(tf.keras.Model):
    
    def __init__(self, hparams, is_train):
        super(UniversalTransformer, self).__init__()
        self.hparams = hparams
        self.is_train = is_train
        self.embedding_layer = EmbeddingSharedWeights(hparams['vocab_size'], hparams['num_units'])
        self.encoder_stack = EncoderStack(hparams, is_train)
        self.encoder_embedding_dropout = tf.keras.layers.Dropout(hparams['dropout_rate'])
        
        self.decoder_stack = DecoderStack(hparams, is_train)
        self.decoder_embedding_dropout = tf.keras.layers.Dropout(hparams['dropout_rate'])
        
        self.global_step = tf.train.get_or_create_global_step()
    
    def call(self, inputs, targets: Optional[np.ndarray] = None):
        attention_bias = model_utils.get_padding_bias(inputs)
        encoder_outputs, enc_ponders, enc_remainders = self._encode(inputs, attention_bias)
        logits, dec_ponders, dec_remainders = self._decode(encoder_outputs, targets, attention_bias)

        if targets is None:
            raise Exception()
        enc_act_loss = tf.reduce_mean(enc_ponders + enc_remainders)
        dec_act_loss = tf.reduce_mean(dec_ponders + dec_remainders)
        act_loss = self.hparams['act_loss_weight'] * (enc_act_loss + dec_act_loss)
        if self.is_train:
            with tf.contrib.summary.record_summaries_every_n_global_steps(10):
                tf.contrib.summary.scalar('summary/ponder_times_enc', tf.reduce_mean(enc_ponders))
                tf.contrib.summary.scalar('summary/ponder_times_dec', tf.reduce_mean(dec_ponders))
            
        return logits, act_loss
        
    def build_graph(self):
        with tf.name_scope('graph'):
            self.is_training_ph = tf.placeholder(name='is_training', shape=(), dtype=bool)
            self.encoder_inputs_ph = tf.placeholder(name='encoder_inputs', shape=[self.hparams['batch_size'], self.hparams['max_length']], dtype=tf.int32)
            self.decoder_inputs_ph = tf.placeholder(name='decoder_inputs', shape=[self.hparams['batch_size'], self.hparams['max_length']], dtype=tf.int32)

            self.logits = self.call(self.encoder_inputs_ph, self.decoder_inputs_ph)
            
            self.loss_op = self.loss(self.encoder_inputs_ph, self.decoder_inputs_ph)
            self.acc_op = self.acc(self.encoder_inputs_ph, self.decoder_inputs_ph)
        
    def save(self, optimizer):
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         model=self,
                                         optimizer_step=self.global_step)
        checkpoint.save(self.hparams['ckpt_path'])
        
    def load(self, optimizer):
        ckpt_path = tf.train.latest_checkpoint(os.path.dirname(self.hparams['ckpt_path']))
        if ckpt_path:
            checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                                     model=self,
                                                     optimizer_step=self.global_step)
            checkpoint.restore(ckpt_path)
            print('restored')
        else:
            print('not restored because no checkpoint found')
    
    def loss(self, inputs, targets):
        pad = tf.to_float(tf.not_equal(targets, 0))
        onehot_targets = tf.one_hot(targets, self.hparams['vocab_size'])
        logits, act_loss = self(inputs, targets)
        cross_ents = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_targets,
            logits=logits
        )
        loss = tf.reduce_sum(cross_ents * pad) / tf.reduce_sum(pad)
        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            tf.contrib.summary.scalar('summary/target_loss', loss)
            tf.contrib.summary.scalar('summary/act_loss', act_loss)
        return loss + act_loss
    
    def acc(self, inputs, targets):
        logits, _ = self(inputs, targets)
        predicted_ids = tf.to_int32(tf.argmax(logits, axis=2))
        correct = tf.equal(predicted_ids, targets)
        pad = tf.to_float(tf.not_equal(targets, 0))
        acc = tf.reduce_sum(tf.to_float(correct) * pad) / (tf.reduce_sum(pad))
        return acc
        
    def grads(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss = self.loss(inputs, targets)
        return tape.gradient(loss, self.variables)
    
    def fit(self, ds, optimizer, writer):
        """ Function to train the model, using the selected optimizer and
            for the desired number of epochs. It also stores the accuracy
            of the model after each epoch.
        """
        for e in range(self.hparams['num_epochs']):
            batch = ds.feed_dict(None, self.hparams['batch_size'], True)
            start = time.time()
            for b in batch:
                inputs, targets = b[0], b[2]
                loss = self.loss(inputs, targets)
                acc = self.acc(inputs, targets)
                
                grads = self.grads(inputs, targets)
                optimizer.apply_gradients(zip(grads, self.variables), self.global_step)
                step = self.global_step.numpy()
                with tf.contrib.summary.record_summaries_every_n_global_steps(10):
                    tf.contrib.summary.scalar('summary/acc', acc)
                    tf.contrib.summary.scalar('summary/loss', loss)
                    tf.contrib.summary.scalar('summary/learning_rate', self.learning_rate())
            print('elapsed: ', time.time() - start)
            self.save(optimizer)
            print('{} epoch finished. now {} step, loss: {:.4f}, acc: {:.4f}'.format(e, step, loss ,acc))
        
    def predict(self, encoder_outputs, bias):
        pass
        
    def _encode(self, inputs, attention_bias):
        embedded_inputs = self.embedding_layer(inputs)
        inputs_padding = model_utils.get_padding(inputs)

        if self.is_train:
            encoder_inputs = self.encoder_embedding_dropout(embedded_inputs)
        return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)
    
    def _decode(self, encoder_outputs, targets, attention_bias):
        decoder_inputs = self.embedding_layer(targets)
        decoder_inputs = tf.pad(decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        # add positional encoding
        length = tf.shape(decoder_inputs)[1]
        decoder_inputs += model_utils.get_position_encoding(length, self.hparams['num_units'])
        
        if self.is_train:
            decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)

        decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(length)
        outputs, dec_ponders, dec_remainders = self.decoder_stack(decoder_inputs, encoder_outputs, decoder_self_attention_bias, attention_bias)
        logits = self.embedding_layer.linear(outputs)
        return logits, dec_ponders, dec_remainders
    
    def learning_rate(self):
        step = tf.to_float(self.global_step)
        rate = tf.minimum(step ** -0.5, step * self.hparams['warmup_steps'] ** -1.5) * self.hparams['num_units'] ** -0.5
        return rate


class ACT(tf.keras.Model):
    
    def __init__(self, batch_size, length, hidden_size):
        super(ACT, self).__init__()
        
        self.halting_probability = tf.zeros((batch_size, length), name='halting_probability')
        self.remainders = tf.zeros((batch_size, length), name="remainder")
        self.n_updates = tf.zeros((batch_size, length), name="n_updates")
        
    def call(self, pondering, halt_threshold):
        # 今現在まだ計算している symbol だけ取ってくるマスク
        still_running = tf.cast(tf.less(self.halting_probability, 1.0), tf.float32)

        # 今回の stepondering で停止する symbol のマスク、halt_threshold を超えているかどうかチェックしている
        new_halted = tf.greater(self.halting_probability + pondering * still_running, halt_threshold)
        new_halted = tf.cast(new_halted, tf.float32) * still_running

        # ここまででも今回のでも停止しないもののマスク
        still_running_now = tf.less_equal(self.halting_probability + pondering * still_running, halt_threshold)
        still_running_now = tf.cast(still_running_now, tf.float32) * still_running

        # まだ停止していない symbol について、停止する確率を更新
        self.halting_probability += pondering * still_running

        # 今回停止した symbol について、remainder の計算して停止確率を更新
        self.remainders += new_halted * (1 - self.halting_probability)
        self.halting_probability += new_halted * self.remainders

        # 今回更新があった symbol について更新回数を足す
        self.n_updates += still_running + new_halted

        # 新しい state をどれだけ output に入れるかの weights を計算し、shape をあわせる
        # ここで既に停止している symbol の係数は 0 になるため、値は変わらない
        update_weights = pondering * still_running + new_halted * self.remainders
        update_weights = tf.expand_dims(update_weights, -1)
        
        return update_weights
    
    def should_continue(self, threshold) -> bool:
        return tf.reduce_any(tf.less(self.halting_probability, threshold))


class EncoderStack(tf.keras.Model):
    
    def __init__(self, hparams, is_train):
        super(EncoderStack, self).__init__()
        self.hparams = hparams
        
        self_attention_layer = SelfAttention(hparams['num_units'], hparams['num_heads'], hparams['dropout_rate'], is_train)
        ffn_layer = FeedForwardNetwork(hparams['num_units'], hparams['num_filter_units'], hparams['dropout_rate'], is_train)
        self.self_attention_wrapper = LayerWrapper(self_attention_layer, hparams['num_units'], hparams['dropout_rate'], is_train)
        self.ffn_wrapper = LayerWrapper(ffn_layer, hparams['num_units'], hparams['dropout_rate'], is_train)
        self.output_norm = LayerNormalization(hparams['num_units'])
        self.pondering_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True, bias_initializer=tf.constant_initializer(1.0))
        
    
    def call(self, encoder_inputs, attention_bias, inputs_padding):
        batch_size, length, hidden_size = tf.unstack(tf.shape(encoder_inputs))
        act = ACT(batch_size, length, hidden_size)
        halt_threshold = 1.0 - hparams['act_epsilon']
        
        state = encoder_inputs
        previous_state = tf.zeros_like(state, name='previous_state')
        for step in range(self.hparams['act_max_step']):
            # 終了条件を確認
            if not act.should_continue(halt_threshold):
                break
            
            # position & timestep encoding
            state += model_utils.get_position_encoding(self.hparams['max_length'], hidden_size)
            state += model_utils.get_timestep_encoding(step, self.hparams['act_max_step'], hidden_size)
            
            # pondering 判断のための特徴を計算
            pondering = self.pondering_layer(state)
            pondering = tf.squeeze(pondering, axis=-1)
            
            # proceed act step
            update_weights = act(pondering, halt_threshold)
            
            state = self.self_attention_wrapper(state, attention_bias)
            state = self.ffn_wrapper(state, inputs_padding)
            
            # ここまでの state と weighted sum を取り、 previous_state を更新
            new_state = (state * update_weights) + (previous_state * (1 - update_weights))
            previous_state = new_state

        return self.output_norm(new_state), act.n_updates, act.remainders

class DecoderStack(tf.keras.Model):
    
    def __init__(self, hparams, is_train):
        super(DecoderStack, self).__init__()
        self.my_layers = []
        
        self.hparams = hparams
        self_attention_layer = SelfAttention(hparams['num_units'], hparams['num_heads'], hparams['dropout_rate'], is_train)
        enc_dec_attention_layer = MultiheadAttention(hparams['num_units'], hparams['num_heads'], hparams['dropout_rate'], is_train)
        ffn_layer = FeedForwardNetwork(hparams['num_units'], hparams['num_filter_units'], hparams['dropout_rate'], is_train)
        self.self_attention_wrapper = LayerWrapper(self_attention_layer, hparams['num_units'], hparams['dropout_rate'], is_train)
        self.enc_dec_attention_wrapper = LayerWrapper(enc_dec_attention_layer, hparams['num_units'], hparams['dropout_rate'], is_train)
        self.ffn_wrapper = LayerWrapper(ffn_layer, hparams['num_units'], hparams['dropout_rate'], is_train)
        self.output_norm = LayerNormalization(hparams['num_units'])
        self.pondering_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True, bias_initializer=tf.constant_initializer(1.0))
    
    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias, attention_bias):
        batch_size, length, hidden_size = tf.unstack(tf.shape(decoder_inputs))
        act = ACT(batch_size, length, hidden_size)
        halt_threshold = 1.0 - hparams['act_epsilon']
        
        state = decoder_inputs
        previous_state = tf.zeros_like(state, name='previous_state')
        for step in range(self.hparams['act_max_step']):
            # position and timestep encoding
            state += model_utils.get_position_encoding(self.hparams['max_length'], hidden_size)
            state += model_utils.get_timestep_encoding(step, self.hparams['act_max_step'], hidden_size)
            
            # pondering 判断のための特徴を計算
            pondering = self.pondering_layer(state)
            pondering = tf.squeeze(pondering, axis=-1)
            
            # proceed act step
            update_weights = act(pondering, halt_threshold)
            
            state = self.self_attention_wrapper(state, decoder_self_attention_bias)
            state = self.enc_dec_attention_wrapper(state, encoder_outputs, attention_bias)
            state = self.ffn_wrapper(state)
            
            # ここまでの state と weighted sum を取り、 previous_state を更新
            new_state = (state * update_weights) + (previous_state * (1 - update_weights))
            previous_state = new_state
            
        return self.output_norm(new_state), act.n_updates, act.remainders

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
process_1 = Process(target=worker,args=(hparams1, 2))

process_0.start()

process_1.start()

worker(hparams1, 1)




