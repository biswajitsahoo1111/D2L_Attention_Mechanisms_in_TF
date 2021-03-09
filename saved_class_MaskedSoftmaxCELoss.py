import tensorflow as tf
from saved_func_sequence_mask import sequence_mask

class MaskedSoftmaxCELoss(tf.keras.losses.Loss):
    """The softmax cross-entropy loss with masks."""
    def __init__(self, valid_len):
        super().__init__(reduction = 'none')
        self.valid_len = valid_len
    
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def call(self, label, pred):
        weights = tf.ones_like(label, dtype = tf.float32)
        weights = sequence_mask(weights, self.valid_len)
        label_one_hot = tf.one_hot(label, depth = pred.shape[-1])
        unweighted_loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True, reduction = 'none')(label_one_hot,
                                                                                                          pred)
        weighted_loss = tf.reduce_mean((unweighted_loss * weights), axis = 1)
        return weighted_loss