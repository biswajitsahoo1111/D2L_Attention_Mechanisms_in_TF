import tensorflow as tf

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return f'/GPU:{i}'
    return f'/CPU:0'