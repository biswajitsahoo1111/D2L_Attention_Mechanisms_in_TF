import tensorflow as tf

def sequence_mask(X, valid_len, value = 0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.shape[1]
    mask = (tf.range(start = 0, limit = maxlen, dtype = tf.float32)[None, :] < tf.cast(valid_len[:, None],
                                                                                       dtype = tf.float32)).numpy()
    X = X.numpy()
    X[~mask] = value
    return tf.constant(X)