import tensorflow as tf
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, num_hiddens, dropout, max_len = 1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Create a long enough `P`
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype = np.float32).reshape(
            -1,1)/np.power(10000, np.arange(0, num_hiddens, 2, dtype = np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        
    def call(self, X, training):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, training = training)