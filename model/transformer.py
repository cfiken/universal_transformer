"""Defines the Transformer model, and its encoder and decoder stacks.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer original model code source: https://github.com/tensorflow/tensor2tensor
"""

import os
import time
from typing import Optional
import tensorflow as tf
import numpy as np

from model.attention import MultiheadAttention, SelfAttention
from model.embedding import EmbeddingSharedWeights
from model.ffn import FeedForwardNetwork
from model.layer_utils import LayerNormalization, LayerWrapper
from model import model_utils


class Transformer(tf.keras.Model):
    
    def __init__(self, hparams, is_train):
        super(Transformer, self).__init__()
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
        encoder_outputs = self._encode(inputs, attention_bias)
        
        if targets is None:
            logits = self._decode(encoder_outputs, targets, attention_bias)
            #raise Exception()
            return logits #self.predict(encoder_outputs, attention_bias)
        else:
            logits = self._decode(encoder_outputs, targets, attention_bias)
            return logits
        
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
        logits = self(inputs, targets)
        cross_ents = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_targets,
            logits=logits
        )
        loss = tf.reduce_sum(cross_ents * pad) / tf.reduce_sum(pad)
        return loss
    
    def acc(self, inputs, targets):
        logits = self(inputs, targets)
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
        embedded_inputs += model_utils.get_position_encoding(self.hparams['max_length'], self.hparams['num_units'])
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
        outputs = self.decoder_stack(decoder_inputs, encoder_outputs, decoder_self_attention_bias, attention_bias)
        logits = self.embedding_layer.linear(outputs)
        return logits
    
    def learning_rate(self):
        step = tf.to_float(self.global_step)
        rate = tf.minimum(step ** -0.5, step * self.hparams['warmup_steps'] ** -1.5) * self.hparams['num_units'] ** -0.5
        return rate


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
            encoder_inputs = ffn_layer(encoder_inputs, inputs_padding)
            
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