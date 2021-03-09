import tensorflow as tf
def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`"""
    X = tf.reshape(X, shape = (-1, num_heads, X.shape[1], X.shape[2]))
    X = tf.transpose(X, perm = (0, 2, 1, 3))
    return tf.reshape(X, shape = (X.shape[0], X.shape[1], -1))