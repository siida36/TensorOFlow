# coding: utf-8
#
# Usage:
#   python examples/multi_layer_seq2seq.py -m train -c configs/multi_layer_seq2seq.ini
#   python examples/multi_layer_seq2seq.py -m eval -c configs/multi_layer_seq2seq.ini
#
# Purpose:
#   Input some sequence, then predict same sequence(+ EOS token).

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from configs.configs import Configs
from data.data import simple_data, through
from utils.early_stopping import EarlyStopper
from utils.monitor import Monitor


def main(args):
  tf.reset_default_graph()

  # process config
  c = Configs(args.config)
  ROOT = os.environ['TENSOROFLOW']
  model_path = '%s/examples/model/multi_layer_seq2seq/model' % ROOT
  PAD = c.const['PAD']
  EOS = c.const['EOS']
  train_step = c.option['train_step']
  max_time = c.option['max_time']
  batch_size = c.option['batch_size']
  vocabulary_size = c.option['vocabulary_size']
  input_embedding_size = c.option['embedding_size']
  hidden_units = c.option['hidden_units']
  layers = c.option['layers']
  datas = []

  # placeholder
  encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
  decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
  decoder_labels = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_labels')

  # embed
  embeddings = tf.Variable(tf.random_uniform([vocabulary_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32, name='embeddings')
  encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
  decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

  # encoder
  encoder_units = hidden_units
  encoder_layers = [tf.contrib.rnn.LSTMCell(size) for size in [encoder_units] * layers]
  encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_layers)
  encoder_output, encoder_final_state = tf.nn.dynamic_rnn(
      encoder_cell, encoder_inputs_embedded,
      dtype=tf.float32, time_major=True
  )
  del encoder_output

  # decoder
  decoder_units = encoder_units
  decoder_layers = [tf.contrib.rnn.LSTMCell(size) for size in [decoder_units] * layers]
  decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_layers)
  decoder_output, decoder_final_state = tf.nn.dynamic_rnn(
      decoder_cell, decoder_inputs_embedded,
      initial_state=encoder_final_state,
      scope="plain_decoder",
      dtype=tf.float32, time_major=True
  )

  decoder_logits = tf.contrib.layers.linear(decoder_output, vocabulary_size)
  decoder_prediction = tf.argmax(decoder_logits, 2) # max_time: axis=0, batch: axis=1, vocab: axis=2

  # optimizer
  stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=tf.one_hot(decoder_labels, depth=vocabulary_size, dtype=tf.float32),
      logits=decoder_logits,
  )

  loss = tf.reduce_mean(stepwise_cross_entropy)
  train_op = tf.train.AdamOptimizer().minimize(loss)
  
  saver = tf.train.Saver()
  with tf.Session() as sess:
    if args.mode == 'train':
      # train
      loss_freq = train_step // 100
      loss_log = []
      loss_suffix = ''
      es = EarlyStopper(max_size=5, edge_threshold=0.1)
      m = Monitor(train_step)
      sess.run(tf.global_variables_initializer())
      for i in range(train_step):
        m.monitor(i, loss_suffix)
        batch_data = through(datas, max_time, batch_size, vocabulary_size)
        feed_dict = {encoder_inputs:batch_data['encoder_inputs'],
                     decoder_inputs:batch_data['decoder_inputs'],
                     decoder_labels:batch_data['decoder_labels']}
        sess.run(fetches=[train_op, loss], feed_dict=feed_dict)
        if i % loss_freq == 0:
          batch_data = through(datas, max_time, batch_size, vocabulary_size)
          feed_dict = {encoder_inputs:batch_data['encoder_inputs'],
                       decoder_inputs:batch_data['decoder_inputs'],
                       decoder_labels:batch_data['decoder_labels']}
          loss_val = sess.run(fetches=loss, feed_dict=feed_dict)
          loss_log.append(loss_val)
          loss_suffix = 'loss: %f' % loss_val
          es_status = es(loss_val)
          if i > train_step // 2 and es_status:
            print('early stopping at step: %d' % i)
            break
      saver.save(sess, model_path)
      print('save at %s' % model_path)
      plt.plot(np.arange(len(loss_log)) * loss_freq, loss_log)
      plt.savefig('%s_loss.png' % model_path)
    elif args.mode == 'eval':
      saver.restore(sess, model_path)
      print('load from %s' % model_path)
    else:
      raise

    # evaluate
    batch_data = through(datas, max_time, batch_size, vocabulary_size)
    feed_dict = {encoder_inputs:batch_data['encoder_inputs'],
                 decoder_inputs:batch_data['decoder_inputs'],
                 decoder_labels:batch_data['decoder_labels']}
    pred = sess.run(fetches=decoder_prediction, feed_dict=feed_dict)
    input_ = batch_data['encoder_inputs']
    loss_val = sess.run(fetches=loss, feed_dict=feed_dict)

    print('input sequences...\n{}'.format(input_))
    print('predict sequences...\n{}'.format(pred))
    print('loss: %f' % loss_val)

  print('finish.')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', '-m', type=str, help='train | eval')
  parser.add_argument('--config', '-c', type=str, help='config file path')
  args = parser.parse_args()
  main(args)
  
