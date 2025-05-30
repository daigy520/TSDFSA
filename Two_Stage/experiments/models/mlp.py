"""稀疏MLP模型"""

import tensorflow as tf


class MLPModel(tf.keras.Model):
  def __init__(
      self,
      layer_sequence=None,
      is_classification=True,
      num_classes=None,
      learning_rate=0.001,
      decay_steps=500,
      decay_rate=0.8,
      alpha=0,
      batch_norm=True,
  ):
    """初始化"""

    super().__init__()

    if batch_norm:
      self.batch_norm_layer = tf.keras.layers.BatchNormalization()
    self.batch_norm = batch_norm

    mlp_sequence = [
        tf.keras.layers.Dense(
            dim, activation=tf.keras.layers.LeakyReLU(alpha=alpha)
        )
        for dim in layer_sequence
    ]
    print(f"mlp_sequence before: {mlp_sequence}")
    self.mlp_model = tf.keras.Sequential(mlp_sequence)
    if is_classification:
      self.mlp_predictor = tf.keras.layers.Dense(
          num_classes, activation="softmax", dtype="float32"
      )
    else:
      self.mlp_predictor = tf.keras.layers.Dense(1, dtype="float32")

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False,
    )
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

  def call(self, inputs, training=False):
    if self.batch_norm:
      inputs = self.batch_norm_layer(inputs, training=training)
    inputs = tf.multiply(inputs, self.selected_features)
    representation = self.mlp_model(inputs)  # other layers
    prediction = self.mlp_predictor(representation)
    return prediction
