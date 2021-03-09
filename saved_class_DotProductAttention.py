import tensorflow as tf
from saved_func_masked_softmax import masked_softmax

class DotProductAttention(tf.keras.layers.Layer):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries) or can be `None`
    def call(self, queries, keys, values, valid_lens, training):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b = True)/tf.math.sqrt(tf.cast(d, dtype = tf.float32))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, training = training), values)