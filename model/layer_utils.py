import tensorflow as tf


class LayerWrapper(tf.keras.Model):

    def __init__(self, layer: tf.keras.Model, hidden_size: int, dropout_rate: float, is_train: bool):
        super().__init__()
        self.is_train = is_train
        self.layer = layer
        self.postprocess_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = LayerNormalization(hidden_size)

    def call(self, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        # layer norm
        y = self.layer_norm(x)
        # apply layer
        y = self.layer(y, *args, **kwargs)

        # dropout
        if self.train:
            y = self.postprocess_dropout(y)
        # residual
        y = x + y
        return y


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = self.add_weight('layer_norm_scale',
                                     [self.hidden_size],
                                     dtype=tf.float32,
                                     initializer=tf.ones_initializer())
        self.bias = self.add_weight('layer_norm_bias',
                                    [self.hidden_size],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

    def call(self, x: tf.Tensor, epsilon: float = 1e-6) -> tf.Tensor:
        x = tf.to_float(x)
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias
