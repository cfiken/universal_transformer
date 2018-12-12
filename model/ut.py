import os
import time
from typing import Optional
import tensorflow as tf
import numpy as np
from model.embedding import EmbeddingSharedWeights
from model.attention import MultiheadAttention, SelfAttention
from model.ffn import FeedForwardNetwork
from model.layer_utils import LayerWrapper, LayerNormalization
from model import model_utils


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
        # mask for not halted yet
        still_running = tf.cast(tf.less(self.halting_probability, 1.0), tf.float32)

        # mask for new halted at this step
        new_halted = tf.greater(self.halting_probability + pondering * still_running, halt_threshold)
        new_halted = tf.cast(new_halted, tf.float32) * still_running

        # update mask for not halted yet and not halted at this step
        still_running_now = tf.less_equal(self.halting_probability + pondering * still_running, halt_threshold)
        still_running_now = tf.cast(still_running_now, tf.float32) * still_running

        # update halting probability
        self.halting_probability += pondering * still_running

        # update remainders and halting probability for ones halted at this step
        self.remainders += new_halted * (1 - self.halting_probability)
        self.halting_probability += new_halted * self.remainders

        # update times
        self.n_updates += still_running + new_halted

        # calc update weights for not halted
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
        halt_threshold = 1.0 - self.hparams['act_epsilon']
        
        state = encoder_inputs
        previous_state = tf.zeros_like(state, name='previous_state')
        for step in range(self.hparams['act_max_step']):
            # judge to continue
            if not act.should_continue(halt_threshold):
                break
            
            # position & timestep encoding
            state += model_utils.get_position_encoding(self.hparams['max_length'], hidden_size)
            state += model_utils.get_timestep_encoding(step, self.hparams['act_max_step'], hidden_size)
            
            # to judge pondering
            pondering = self.pondering_layer(state)
            pondering = tf.squeeze(pondering, axis=-1)
            
            # proceed act step
            update_weights = act(pondering, halt_threshold)
            
            state = self.self_attention_wrapper(state, attention_bias)
            state = self.ffn_wrapper(state, inputs_padding)
            
            # update new state and previous state
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
        halt_threshold = 1.0 - self.hparams['act_epsilon']
        
        state = decoder_inputs
        previous_state = tf.zeros_like(state, name='previous_state')
        for step in range(self.hparams['act_max_step']):
            # judge to continue
            if not act.should_continue(halt_threshold):
                break

            # position and timestep encoding
            state += model_utils.get_position_encoding(self.hparams['max_length'], hidden_size)
            state += model_utils.get_timestep_encoding(step, self.hparams['act_max_step'], hidden_size)
            
            # to judge pondering
            pondering = self.pondering_layer(state)
            pondering = tf.squeeze(pondering, axis=-1)
            
            # proceed act step
            update_weights = act(pondering, halt_threshold)
            
            state = self.self_attention_wrapper(state, decoder_self_attention_bias)
            state = self.enc_dec_attention_wrapper(state, encoder_outputs, attention_bias)
            state = self.ffn_wrapper(state)
            
            # update new state and previous state
            new_state = (state * update_weights) + (previous_state * (1 - update_weights))
            previous_state = new_state
            
        return self.output_norm(new_state), act.n_updates, act.remainders