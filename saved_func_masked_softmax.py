import tensorflow as tf
from saved_func_sequence_mask import sequence_mask

def masked_softmax(X, valid_lens):
    """X: 3D tensor
       valid_lens: 1D or 2D tensor
       When valid_lens is 1D, its `len(valid_lens)` should be same as `X.shape[0]`.
       When valid_lens is 2D, `valid_lens.shape` should be same as `X.shape[:2]`.
    """
    if valid_lens is None:
        return tf.nn.softmax(X, axis = -1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            if (len(valid_lens) == X.shape[0]): 
                valid_lens = tf.repeat(valid_lens, repeats = shape[1])
            else:
                print("Valid_lens shape is incompatible with input. Read docstring of `masked_softmax` function.")
        else:
            if (valid_lens.shape == X.shape[:2]):
                valid_lens = tf.reshape(valid_lens, shape = -1)
            else:
                print("Valid_lens shape is incompatible with input. Read docstring of `masked_softmax` function.")
            
        
        X = sequence_mask(tf.reshape(X, shape = (-1, shape[-1])), valid_lens, value=-1e6)    
        return tf.nn.softmax(tf.reshape(X, shape = shape), axis = -1)