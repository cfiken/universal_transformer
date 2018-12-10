import tensorflow as tf


class FeedForwardNetwork(tf.keras.Model):

    def __init__(self, hidden_size: int, filter_size: int, dropout_rate: float, is_train: bool):
        super(FeedForwardNetwork, self).__init__()
        self.is_train = is_train
        self.hidden_size = hidden_size
        self.filter_dense_layer = tf.keras.layers.Dense(filter_size, tf.nn.relu, use_bias=True, name='filter_layer')
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate, name='relu_dropout')
        self.output_dense_layer = tf.keras.layers.Dense(hidden_size, use_bias=True, name='output_layer')

    def call(self, x: tf.Tensor, padding=None) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        if padding is not None:
            # removing pad
            pad_mask = tf.reshape(padding, [-1])
            nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

            x = tf.reshape(x, [-1, self.hidden_size])
            x = tf.gather_nd(x, indices=nonpad_ids)

            x.set_shape([None, self.hidden_size])
            x = tf.expand_dims(x, axis=0)  # [1, batch_size x length, hidden_size]

        outputs = self.filter_dense_layer(x)
        if self.is_train:
            outputs = self.dropout_layer(outputs)
        outputs = self.output_dense_layer(outputs)

        if padding is not None:
            # re add padding
            outputs = tf.squeeze(outputs, axis=0)  # [batch_size x length, hidden_units]
            outputs = tf.scatter_nd(
                indices=nonpad_ids,
                updates=outputs,
                shape=[batch_size * length, self.hidden_size]
            )
            outputs = tf.reshape(outputs, [batch_size, length, self.hidden_size])

        return outputs
