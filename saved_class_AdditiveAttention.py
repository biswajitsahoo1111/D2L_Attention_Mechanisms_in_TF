import tensorflow as tf
from saved_func_masked_softmax import masked_softmax
class AdditiveAttention(tf.keras.layers.Layer):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias = False)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias = False)
        self.w_v = tf.keras.layers.Dense(1, use_bias = False)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, queries, keys, values, valid_lens, training):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of `queries`: (`batch_size`, no. of
        # queries, 1, `num_hiddens`) and shape of `keys`: (`batch_size`, 1,
        # no. of key-value pairs, `num_hiddens`). Sum them up with
        # broadcasting
        features = tf.expand_dims(queries, axis = 2) + tf.expand_dims(keys, axis = 1)
        features = tf.nn.tanh(features)
        # There is only one output of `self.w_v`, so we remove the last
        # one-dimensional entry from the shape. Shape of `scores`:
        # (`batch_size`, no. of queries, no. of key-value pairs)
        scores = tf.squeeze(self.w_v(features), axis = -1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of `values`: (`batch_size`, no. of key-value pairs, value
        # dimension)
        return tf.matmul(self.dropout(self.attention_weights, training = training), values)