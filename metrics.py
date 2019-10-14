import tensorflow as tf
import keras.backend as K

def focalLoss(target, y_hat):
  """ if y = 1, p = p, alpha = alpha
  else if y = 0, p = 1 - p, alpha = 1 - alpha
  form: -alpha * (1-p)^gamma * log(p) """
  gamma = 2.0
  alpha = 0.25

  zeros = tf.equal(target, 0)
  # alpha = tf.constant(alpha, shape = tf.shape(target))  
  alpha = tf.fill(tf.shape(target), alpha)
  alpha = tf.where(zeros, 1-alpha, alpha)

  y_hat = tf.cast(y_hat, tf.float32)
  target = tf.cast(target, tf.float32)

  # jitter data to range (0,1)
  zeros = tf.equal(y_hat, 0)
  y_hat = tf.where(zeros, y_hat+K.epsilon(), y_hat)
  ones = tf.equal(y_hat, 1)
  y_hat = tf.where(ones, y_hat-K.epsilon(), y_hat)

  relative_target_probs = tf.where(zeros, 1-y_hat, y_hat)
  loss = (-alpha * (1-relative_target_probs)**gamma) * tf.log(relative_target_probs)
  return K.mean(K.sum(loss, axis = 1))

