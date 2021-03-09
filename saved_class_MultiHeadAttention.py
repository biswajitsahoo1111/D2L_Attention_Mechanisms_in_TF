import tensorflow as tf
from saved_class_DotProductAttention import DotProductAttention
from saved_func_transpose_qkv import transpose_qkv
from saved_func_transpose_output import transpose_output

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias = False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias = bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias = bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias = bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias = bias)
    
    def call(self, queries, keys, values, valid_lens, training):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        
        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = tf.repeat(valid_lens, repeats = self.num_heads, axis = 0)
            
        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries, `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens, training = training)
        
        # Shape of `output_concat`: (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)