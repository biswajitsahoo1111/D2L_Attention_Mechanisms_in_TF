import tensorflow as tf

def sequence_mask(X, valid_len, value = 0):
    """Mask irrelevant entries in sequences.
    Input argument X of this function is 2D.
    """
    maxlen = X.shape[1]
    mask = tf.range(start = 0, limit = maxlen, dtype = tf.float32)[None, :] < tf.cast(valid_len[:, None], dtype = tf.float32)
    return tf.where(mask, X, value)