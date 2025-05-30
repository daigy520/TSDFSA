# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Feature selection with Orthogonal Matching Pursuit."""

from Two_Stage.benchmarks.orthogonal_matching_pursuit import OrthogonalMatchingPursuit
from Two_Stage.experiments.models.mlp import MLPModel
import tensorflow as tf


class OrthogonalMatchingPursuitModel(MLPModel):
  """MLP with Orthogonal Matching Pursuit."""

  def __init__(
      self,
      num_inputs,
      num_inputs_to_select,
      num_train_steps,
      num_inputs_to_select_per_step=1,
      **kwargs,
  ):
    """Initialize the model."""

    super(OrthogonalMatchingPursuitModel, self).__init__(**kwargs)

    self.omp = OrthogonalMatchingPursuit(
        num_inputs,
        num_inputs_to_select,
        num_inputs_to_select_per_step=num_inputs_to_select_per_step,
    )

    self.num_inputs = num_inputs
    self.num_train_steps = num_train_steps

  def call(self, inputs, training=False, omp_attention=True):
    training_percentage = self.optimizer.iterations / self.num_train_steps
    if self.batch_norm:
      inputs = self.batch_norm_layer(inputs, training=training)
    if omp_attention:
      feature_weights = self.omp(training_percentage)
    else:
      feature_weights = tf.ones(self.num_inputs)
    inputs = tf.multiply(inputs, feature_weights)
    representation = self.mlp_model(inputs)
    prediction = self.mlp_predictor(representation)
    return prediction

  def train_step(self, inputs):
    """Custom train step for OMP with correct gradient update."""
    x, y = inputs
    training_percentage = self.optimizer.iterations / self.num_train_steps

    # --------- 正常前向（带当前OMP特征掩码）计算 loss ---------
    with tf.GradientTape() as tape:
        y_pred = self.call(x, training=True)  # 这里默认 omp_attention=True
        loss = self.compute_loss(x, y, y_pred)

    # 计算用于正常训练的梯度
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # --------- 单独计算 OMP特征选择用的梯度 ---------
    with tf.GradientTape() as tape:
        feature_weights = tf.ones(self.num_inputs)  # 全1，不屏蔽任何特征
        x_unmasked = tf.multiply(x, feature_weights)
        if self.batch_norm:
            x_unmasked = self.batch_norm_layer(x_unmasked, training=True)
        representation = self.mlp_model(x_unmasked)
        y_pred_unmasked = self.mlp_predictor(representation)
        omp_loss = self.compute_loss(x, y, y_pred_unmasked)

    # 只对第一层权重计算特征梯度
    first_layer_weights = self.mlp_model.weights[0]
    gradients = tape.gradient(omp_loss, first_layer_weights)
    gradients = tf.norm(gradients, axis=1)  # 取每个输入特征对应的梯度L2范数

    # 更新 OMP 内部的梯度缓存
    assign_gradient = self.omp.gradients.assign(gradients)

    # --------- 更新metrics ---------
    with tf.control_dependencies([assign_gradient]):  # 强制执行assign
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
