import tensorflow as tf

def sequence_mask(X, valid_len, value = 0):
    """Mask irrelevant entries in sequences.
    Argument:
        X: either a 2D or 3D tensor
        valid_len: 1D tensor
        value: value to be substitued for mask
    """
    maxlen = X.shape[1]
    mask = tf.range(start = 0, limit = maxlen, dtype = tf.float32)[None, :] < tf.cast(valid_len[:, None], dtype = tf.float32)
    
    if len(X.shape) == 3:
        return tf.where(tf.expand_dims(mask, axis = -1), X, value)
    else:
        return tf.where(mask, X, value)