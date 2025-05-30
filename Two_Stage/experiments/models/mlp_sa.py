"""选择特征"""

from Two_Stage.experiments.models.mlp import MLPModel
from Two_Stage.sequential_attention import SequentialAttention
import tensorflow as tf


class SequentialAttentionModel(MLPModel):

  def __init__(
      self,
      num_inputs,
      num_inputs_to_select,
      num_train_steps,
      num_inputs_to_select_per_step=1,
      **kwargs,
  ):
    """初始化"""

    super(SequentialAttentionModel, self).__init__(**kwargs)
    self.seqatt = SequentialAttention(
        num_candidates=num_inputs,
        num_candidates_to_select=num_inputs_to_select,
        num_candidates_to_select_per_step=num_inputs_to_select_per_step,
    )
    self.num_train_steps = num_train_steps

  def call(self, inputs, training=False):
    if self.batch_norm:
      inputs = self.batch_norm_layer(inputs, training=training)
    training_percentage = self.optimizer.iterations / self.num_train_steps
    feature_weights = self.seqatt(training_percentage)
    #inputs = tf.multiply(inputs, feature_weights)
    inputs = tf.add(inputs, feature_weights)
    representation = self.mlp_model(inputs)  # other layers
    prediction = self.mlp_predictor(representation)
    return prediction
