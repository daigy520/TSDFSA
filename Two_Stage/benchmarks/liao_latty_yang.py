import tensorflow as tf


class LiaoLattyYangMask(tf.Module):
  """基于注意力机制的特征选择算法"""

  def __init__(self, num_inputs, name='liao_latty_yang_mask', **kwargs):
    super(LiaoLattyYangMask, self).__init__(name=name, **kwargs)

    mlp_sequence = [
        tf.keras.layers.Dense(
            dim, activation=tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        for dim in [128, 64, num_inputs]
    ]
    self.mlp_model = tf.keras.Sequential(mlp_sequence)

  def __call__(self, inputs):
    nonlinear = self.mlp_model(inputs)
    batch_size = tf.cast(tf.shape(inputs)[0], tf.float32)
    logits = tf.reduce_sum(nonlinear, axis=0) / batch_size
    return tf.nn.softmax(logits)
