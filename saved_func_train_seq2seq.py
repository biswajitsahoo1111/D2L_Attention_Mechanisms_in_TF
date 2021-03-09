import tensorflow as tf
from saved_class_Animator import Animator
from saved_class_Timer import Timer
from saved_class_Accumulator import Accumulator
from saved_class_MaskedSoftmaxCELoss import MaskedSoftmaxCELoss
from saved_func_grad_clipping import grad_clipping

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    animator = Animator(xlabel = "epoch", ylabel = "loss", xlim = [10, num_epochs])
    
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)   # Sum of training loss, no. of tokens
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x for x in batch]
            bos = tf.reshape(tf.constant([tgt_vocab['<bos>']] * Y.shape[0]), shape = (-1, 1))
            dec_input = tf.concat([bos, Y[:, :-1]], 1)   # Teacher forcing
            with tf.GradientTape() as tape:
                Y_hat, _ = net(X, dec_input, X_valid_len, training = True)
                l = MaskedSoftmaxCELoss(Y_valid_len)(Y, Y_hat)
            gradients = tape.gradient(l, net.trainable_variables)
            gradients = grad_clipping(gradients, 1)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))
            num_tokens = tf.reduce_sum(Y_valid_len).numpy()
            metric.add(tf.reduce_sum(l), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} ' f'tokens/sec on {str(device)}')