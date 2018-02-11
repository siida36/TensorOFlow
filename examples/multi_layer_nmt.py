# coding: utf-8
#
# Usage:
#   python examples/multi_layer_nmt.py -m train -c configs/multi_layer_nmt.ini
#   python examples/multi_layer_nmt.py -m eval -c configs/multi_layer_nmt.ini
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
from data.data import read_data, read_words, batchnize, build_dictionary, sentence_to_onehot, seq2seq
from utils.early_stopping import EarlyStopper
from utils.monitor import Monitor


def main(args):
  # process config
  c = Configs(args.config)
  ROOT = os.environ['TENSOROFLOW']
  model_path = '%s/examples/model/multi_layer_nmt/model' % ROOT
  PAD = c.const['PAD']
  EOS = c.const['EOS']
  train_step = c.option['train_step']
  max_time = c.option['max_time']
  batch_size = c.option['batch_size']
  vocabulary_size = c.option['vocabulary_size']
  input_embedding_size = c.option['embedding_size']
  hidden_units = c.option['hidden_units']
  layers = c.option['layers']
  source_train_data_path = c.data['source_train_data']
  target_train_data_path = c.data['target_train_data']
  source_valid_data_path = c.data['source_valid_data']
  target_valid_data_path = c.data['target_valid_data']
  source_test_data_path = c.data['source_test_data']
  target_test_data_path = c.data['target_test_data']

  # read data
  source_dictionary, source_reverse_dictionary = build_dictionary(read_words(source_train_data_path), vocabulary_size)
  source_train_datas = [sentence_to_onehot(lines, source_dictionary) for lines in read_data(source_train_data_path)]
  target_dictionary, target_reverse_dictionary = build_dictionary(read_words(target_train_data_path), vocabulary_size)
  target_train_datas = [sentence_to_onehot(lines, target_dictionary) for lines in read_data(target_train_data_path)]

  source_valid_datas = [sentence_to_onehot(lines, source_dictionary) for lines in read_data(source_valid_data_path)]
  target_valid_datas = [sentence_to_onehot(lines, target_dictionary) for lines in read_data(target_valid_data_path)]
  source_test_datas = [sentence_to_onehot(lines, source_dictionary) for lines in read_data(source_test_data_path)]
  target_test_datas = [sentence_to_onehot(lines, target_dictionary) for lines in read_data(target_test_data_path)]

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
  batch_idx = {'train': 0, 'valid': 0, 'test': 0}
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
        source_train_batch, _ = batchnize(source_train_datas, batch_size, batch_idx['train'])
        target_train_batch, batch_idx['train'] = batchnize(target_train_datas, batch_size, batch_idx['train'])
        batch_data = seq2seq(source_train_batch, target_train_batch, max_time, vocabulary_size)
        feed_dict = {encoder_inputs:batch_data['encoder_inputs'],
                     decoder_inputs:batch_data['decoder_inputs'],
                     decoder_labels:batch_data['decoder_labels']}
        sess.run(fetches=[train_op, loss], feed_dict=feed_dict)
        if i % loss_freq == 0:
          source_valid_batch, _ = batchnize(source_valid_datas, batch_size, batch_idx['valid'])
          target_valid_batch, batch_idx['valid'] = batchnize(target_valid_datas, batch_size, batch_idx['valid'])
          batch_data = seq2seq(source_valid_batch, target_valid_batch, max_time, vocabulary_size)
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
    loss_val = []
    input_vectors = None
    predict_vectors = None
    for i in range(len(source_test_datas) // batch_size + 1):
      source_test_batch, _ = batchnize(source_test_datas, batch_size, batch_idx['test'])
      target_test_batch, batch_idx['test'] = batchnize(target_test_datas, batch_size, batch_idx['test'])
      batch_data = seq2seq(source_test_batch, target_test_batch, max_time, vocabulary_size)
      feed_dict = {encoder_inputs:batch_data['encoder_inputs'],
                   decoder_inputs:batch_data['decoder_inputs'],
                   decoder_labels:batch_data['decoder_labels']}
      pred = sess.run(fetches=decoder_prediction, feed_dict=feed_dict)
      if predict_vectors is None:
        predict_vectors = pred.T
      else:
        predict_vectors = np.vstack((predict_vectors, pred.T))
      input_ = batch_data['encoder_inputs']
      if input_vectors is None:
        input_vectors = input_.T
      else:
        input_vectors = np.vstack((input_vectors, input_.T))
      loss_val.append(sess.run(fetches=loss, feed_dict=feed_dict))

    input_sentences = ''
    predict_sentences = ''
    for i, (input_vector, predict_vector) in enumerate(zip(input_vectors[:len(source_test_datas)], predict_vectors[:len(target_test_datas)])):
      input_sentences += ' '.join([source_reverse_dictionary[vector] for vector in input_vector if not vector == PAD])
      predict_sentences += ' '.join([target_reverse_dictionary[vector] for vector in predict_vector if not vector == PAD])
      if i < len(source_test_datas) - 1:
        input_sentences += '\n'
        predict_sentences += '\n'

    evaluate_input_path = '%s.evaluate_input' % model_path
    evaluate_predict_path = '%s.evaluate_predict' % model_path
    with open(evaluate_input_path, 'w') as f1, \
         open(evaluate_predict_path, 'w') as f2:
      f1.write(input_sentences)
      f2.write(predict_sentences)

    print('input sequences at {}'.format(evaluate_input_path))
    print('predict sequences at {}'.format(evaluate_predict_path))
    print('mean of loss: %f' % np.mean(loss_val))

  print('finish.')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', '-m', type=str, help='train | eval')
  parser.add_argument('--config', '-c', type=str, help='config file path')
  args = parser.parse_args()
  main(args)
  
