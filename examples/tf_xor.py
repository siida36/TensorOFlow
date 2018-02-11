import argparse

import tensorflow as tf
import numpy as np

def main(args):
  input_dim=2
  hidden_dim=2
  output_dim=1

  x = tf.placeholder('float', [None, input_dim])
  W1 = tf.Variable(tf.random_uniform([input_dim, hidden_dim], -1.0, 1.0))
  b1 = tf.Variable(tf.random_normal([hidden_dim]))
  W2 = tf.Variable(tf.random_uniform([hidden_dim, output_dim], -1.0, 1.0))
  b2 = tf.Variable(tf.random_normal([output_dim]))
  layer1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
  layer2 = tf.nn.sigmoid(tf.matmul(layer1, W2) + b2)
  y = layer2

  y_ = tf.placeholder('float', [None, output_dim])
  loss = tf.reduce_mean(tf.square(y - y_))

  train_step = tf.train.MomentumOptimizer(0.01, 0.97).minimize(loss)

  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  batch_xs = np.array([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]])
  batch_ys = np.array([
    [0.],
    [1.],
    [1.],
    [0.]])

  if args.mode == 'train':
    for i in range(5000):
      sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
      if i % 1000 == 0:
        print(i, sess.run(y, feed_dict={x: batch_xs, y_:batch_ys}))

    saver = tf.train.Saver()
    saver.save(sess, "model/tf_xor/model")
  else:
    saver = tf.train.Saver()
    saver.restore(sess, "model/tf_xor/model")
    
  sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
  print(sess.run(y, feed_dict={x: batch_xs, y_:batch_ys}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', '-m', type=str, help='train | eval')
  args = parser.parse_args()
  main(args)
