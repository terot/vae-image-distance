import tensorflow as tf
import edward as ed

from tensorflow.contrib import slim
from edward.models import Bernoulli, Normal

class Model:
  def __init__(self, batch_size, dim_latent):
    image_width = 28
    image_height = 28

    # Model
    z = Normal(loc=tf.zeros([batch_size, dim_latent]),
               scale=tf.ones([batch_size, dim_latent]))
    logits = generative_network(z, batch_size, dim_latent)
    x = Bernoulli(logits=logits)

    # Inference
    self.x_ph = tf.placeholder(tf.int32, [batch_size, image_width * image_height])
    self.loc, self.scale = inference_network(tf.cast(self.x_ph, tf.float32), batch_size, dim_latent)
    self.qz = Normal(loc=self.loc, scale=self.scale)

    # Bind p(x, z) and q(z | x) to the same placeholder for x.
    data = {x: self.x_ph}
    self.inference = ed.KLqp({z: self.qz}, data)
    optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
    self.inference.initialize(optimizer=optimizer)

    self.hidden_rep = tf.sigmoid(logits)


def generative_network(z, M, d):
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.

  logits = neural_network(z)
  """
  with slim.arg_scope([slim.conv2d_transpose],
                      activation_fn=tf.nn.elu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'scale': True}):
    net = tf.reshape(z, [M, 1, 1, d])
    net = slim.conv2d_transpose(net, 128, 3, padding='VALID')
    net = slim.conv2d_transpose(net, 64, 5, padding='VALID')
    net = slim.conv2d_transpose(net, 32, 5, stride=2)
    net = slim.conv2d_transpose(net, 1, 5, stride=2, activation_fn=None)
    net = slim.flatten(net)
    return net


def inference_network(x, M, d):
  """Inference network to parameterize variational model. It takes
  data as input and outputs the variational parameters.

  loc, scale = neural_network(x)
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.elu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'scale': True}):
    net = tf.reshape(x, [M, 28, 28, 1])
    net = slim.conv2d(net, 32, 5, stride=2)
    net = slim.conv2d(net, 64, 5, stride=2)
    net = slim.conv2d(net, 128, 5, padding='VALID')
    net = slim.dropout(net, 0.9)
    net = slim.flatten(net)
    params = slim.fully_connected(net, d * 2, activation_fn=None)

  loc = params[:, :d]
  scale = tf.nn.softplus(params[:, d:])
  return loc, scale

