import tensorflow as tf

"""
    Gradient accumulation implementation
"""

def accumulated_gradients(gradients,
                          step_gradients,
                          num_grad_accumulates) -> tf.Tensor:
    if gradients is None:
        gradients = [flat_gradients(g) / num_grad_accumulates for g in step_gradients]
    else:
        for i, g in enumerate(step_gradients):
            gradients[i] += flat_gradients(g) / num_grad_accumulates
    
    return gradients

# This is needed for tf.gather like operations.
def flat_gradients(grads_or_idx_slices: tf.Tensor) -> tf.Tensor:
    '''Convert gradients if it's tf.IndexedSlices.
    When computing gradients for operation concerning `tf.gather`, the type of gradients 
    '''
    if type(grads_or_idx_slices) == tf.IndexedSlices:
        return tf.scatter_nd(
            tf.expand_dims(grads_or_idx_slices.indices, 1),
            grads_or_idx_slices.values,
            grads_or_idx_slices.dense_shape
        )
    return grads_or_idx_slices


"""
    Custom scheduler with learning rate rising for warmup period and going down later on
"""

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, warmup_steps, num_steps, base_lr):
    super(CustomSchedule, self).__init__()

    self.warmup_steps = tf.cast(warmup_steps, tf.float32)
    self.num_steps = tf.cast(num_steps, tf.float32)
    self.lr = tf.cast(base_lr, tf.float32)

  def __call__(self, step):
    def warmupPhase() : return step/tf.math.maximum(1.0, self.warmup_steps)
    def decayPhase() : return tf.math.maximum(0.0, (self.num_steps - step))/tf.math.maximum(1.0, self.num_steps - self.warmup_steps)

    multiplier = tf.cond(tf.math.less(step, self.warmup_steps), warmupPhase, decayPhase)
    
    return self.lr * multiplier