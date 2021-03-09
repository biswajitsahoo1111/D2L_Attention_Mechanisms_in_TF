import tensorflow as tf
def transpose_qkv(X, num_heads):
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`, `num_hiddens` / `num_heads`)
    X = tf.reshape(X, shape = (X.shape[0], X.shape[1], num_heads, -1))

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = tf.transpose(X, perm = (0, 2, 1, 3))

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs, `num_hiddens` / `num_heads`)
    return tf.reshape(X, shape = (-1, X.shape[2], X.shape[3]))