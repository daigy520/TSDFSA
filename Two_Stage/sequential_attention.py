import tensorflow as tf


class SequentialAttention(tf.Module):

  def __init__(
      self,
      num_candidates,
      num_candidates_to_select,
      num_candidates_to_select_per_step=1,
      start_percentage=0.1,
      stop_percentage=1.0,
      name='Two_Stage',
      reset_weights=True,
      **kwargs,
  ):

    super(SequentialAttention, self).__init__(name=name, **kwargs)

    assert num_candidates_to_select % num_candidates_to_select_per_step == 0, (
        'num_candidates_to_select must be a multiple of '
        'num_candidates_to_select_per_step.'
    )

    with self.name_scope:
      self._num_candidates = num_candidates
      self._num_candidates_to_select_per_step = (
          num_candidates_to_select_per_step
      )
      self._num_steps = (
          num_candidates_to_select // num_candidates_to_select_per_step
      )
      self._start_percentage = start_percentage
      self._stop_percentage = stop_percentage
      self._reset_weights = reset_weights

      init_attention_weights = tf.random.normal(
          shape=[num_candidates], stddev=0.00001, dtype=tf.float32
      )
      self._attention_weights = tf.Variable(
          initial_value=lambda: init_attention_weights,
          dtype=tf.float32,
          name='attention_weights',
      )

      self.selected_features = tf.Variable(
          tf.zeros(shape=[num_candidates], dtype=tf.float32),
          trainable=False,
          name='selected_features',
      )

  @tf.Module.with_name_scope
  def __call__(self, training_percentage):
    """计算未选特征权重
    training_percentage： 已完成训练过程的百分比。该输入参数应介于 0 和 1 之间，且应单调递增。
    通过start_percentage参数设置预训练划分占比，将训练过程从线性过程映射成一个非线性过程
    返回值：与未选特征维度一致的注意力权重向量。所有权重介于 0 和 1 之间，且总和为 1。
    """
    percentage = (training_percentage - self._start_percentage) / (
        self._stop_percentage - self._start_percentage
    )
    curr_index = tf.cast(
        tf.math.floor(percentage * self._num_steps), dtype=tf.float32
    )
    curr_index = tf.math.minimum(curr_index, self._num_steps - 1.0)

    should_train = tf.less(curr_index, 0.0)

    num_selected = tf.math.reduce_sum(self.selected_features)
    should_select = tf.greater_equal(curr_index, num_selected)
    _, new_indices = tf.math.top_k(
        self._softmax_with_mask(
            self._attention_weights, 1.0 - self.selected_features
        ),
        k=self._num_candidates_to_select_per_step,
    )
    new_indices = self._k_hot_mask(new_indices, self._num_candidates)
    new_indices = tf.cond(
        should_select,
        lambda: new_indices,
        lambda: tf.zeros(self._num_candidates),
    )
    select_op = self.selected_features.assign_add(new_indices)
    init_attention_weights = tf.random.normal(
        shape=[self._num_candidates], stddev=0.00001, dtype=tf.float32
    )
    should_reset = tf.logical_and(should_select, self._reset_weights)
    new_weights = tf.cond(
        should_reset,
        lambda: init_attention_weights,
        lambda: self._attention_weights,
    )
    reset_op = self._attention_weights.assign(new_weights)

    with tf.control_dependencies([select_op, reset_op]):
      candidates = 1.0 - self.selected_features
      #默认使用softmax调整未选特征权重，更改函数名
      softmax = self._softmax_with_mask(self._attention_weights, candidates)

      return tf.cond(
          should_train,
          lambda: tf.ones(self._num_candidates),
          lambda: softmax + self.selected_features,
      )
  @tf.Module.with_name_scope
  def _k_hot_mask(self, indices, depth, dtype=tf.float32):
    return tf.math.reduce_sum(tf.one_hot(indices, depth, dtype=dtype), 0)

  @tf.Module.with_name_scope
  def _softmax_with_mask(self, logits, mask):
    shifted_logits = logits - tf.math.reduce_max(logits)
    exp_shifted_logits = tf.math.exp(shifted_logits)
    masked_exp_shifted_logits = tf.multiply(exp_shifted_logits, mask)
    return tf.math.divide_no_nan(
        masked_exp_shifted_logits, tf.math.reduce_sum(masked_exp_shifted_logits)
    )

  @tf.Module.with_name_scope
  def _l1_norm(self, attention_weights, mask):
        # L1 正则化，计算L1范数的惩罚项
        l1_norm = tf.reduce_sum(tf.abs(attention_weights))  # 计算L1范数
        return l1_norm  # 将L1正则化项加到权重上

  @tf.Module.with_name_scope
  def _l2_norm(self, attention_weights, mask):
        # L2 正则化，计算L2范数的惩罚项
        l2_norm = tf.reduce_sum(tf.square(attention_weights))  # 计算L2范数
        return l2_norm

  @tf.Module.with_name_scope
  def _l1_normalized(self, attention_weights, mask):
        """L1 范式归一化"""
        l1_norm = tf.reduce_sum(tf.abs(attention_weights), axis=-1, keepdims=True)  # 计算L1范数
        normalized_weights = attention_weights / l1_norm  # 归一化
        return normalized_weights

  @tf.Module.with_name_scope
  def _l2_normalized(self, attention_weights, mask):
        l2_norm = tf.reduce_sum(tf.square(attention_weights), axis=-1, keepdims=True)  # 计算L2范数
        normalized_weights = attention_weights / tf.sqrt(l2_norm)  # 归一化
        return normalized_weights