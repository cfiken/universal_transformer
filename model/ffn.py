import tensorflow as tf


class FeedForwardNetwork(tf.keras.Model):

    def __init__(self, hidden_size: int, filter_size: int, dropout_rate: float, is_train: bool):
        super().__init__()
        self.is_train = is_train
        self.filter_dense_layer = tf.keras.layers.Dense(filter_size, tf.nn.relu, use_bias=True, name='filter_layer')
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate, 'relu_dropout')
        self.output_dense_layer = tf.keras.layers.Dense(hidden_size, use_bias=True, name='output_layer')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        outputs = self.filter_dense_layer(x)
        if self.is_train:
            outputs = self.dropout_layer(outputs)
        outputs = self.output_dense_layer(outputs)
        return outputs
