# coding: utf-8
#
# Usage:
#   python examples/chainer_attention_seq2seq.py -m train -c configs/chainer_attention_seq2seq.ini
#   python examples/chainer_attention_seq2seq.py -m eval -c configs/chainer_attention_seq2seq.ini
#
# Purpose:
#   Input some sequence, then predict same sequence(+ EOS token).

import argparse
import os
import pickle
import pathlib
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from chainer import cuda, optimizer, optimizers, serializers

from configs.configs import Configs
from data.data import read_data, read_words, batchnize, build_dictionary, sentence_to_onehot, seq2seq
from networks.attention_seq2seq import LSTM_Encoder, LSTM_Decoder, Seq2Seq
from utils.early_stopping import EarlyStopper
from utils.monitor import Monitor
from utils.logger import Logger


def main(args):
  # process config
  c = Configs(args.config)
  ROOT = os.environ['TENSOROFLOW']
  output = c.option.get('output', 'examples/model/buf')
  model_directory = '%s/%s' % (ROOT, output)
  model_path = '%s/model' % model_directory
  dictionary_path = {'source': '%s/source_dictionary.pickle' % model_directory,
                     'source_reverse': '%s/source_reverse_dictionary.pickle' % model_directory,
                     'target': '%s/target_dictionary.pickle' % model_directory,
                     'target_reverse': '%s/target_reverse_dictionary.pickle' % model_directory }
  PAD = c.const['PAD']
  BOS = c.const['BOS']
  EOS = c.const['EOS']
  train_step = c.option['train_step']
  max_time = c.option['max_time']
  batch_size = c.option['batch_size']
  vocabulary_size = c.option['vocabulary_size']
  embedding_size = c.option['embedding_size']
  hidden_size = c.option['hidden_size']
  layers = c.option['layers']
  source_train_data_path = c.data['source_train_data']
  target_train_data_path = c.data['target_train_data']
  source_valid_data_path = c.data['source_valid_data']
  target_valid_data_path = c.data['target_valid_data']
  source_test_data_path = c.data['source_test_data']
  target_test_data_path = c.data['target_test_data']

  # initialize output directory
  if pathlib.Path(model_directory).exists():
    if args.overwrite:
      shutil.rmtree(model_directory)
      print('Old model was overwritten.')
    else:
      print('Warning: model %s is exists.')
      print('Old model will be overwritten.')
      while True:
        print('Do you wanna continue? [yes|no]')
        command = input('> ')
        if command == 'yes':
          shutil.rmtree(model_directory)
          print('Old model was overwritten.')
          break
        elif command == 'no':
          sys.exit()
        else:
          print('You can only input "yes" or "no".')

  print('Make new model: %s' % model_directory)
  pathlib.Path(model_directory).mkdir()

  # read data
  if args.mode == 'train':
    source_dictionary, source_reverse_dictionary = build_dictionary(read_words(source_train_data_path), vocabulary_size)
    source_train_datas = [sentence_to_onehot(lines, source_dictionary) for lines in read_data(source_train_data_path)]
    target_dictionary, target_reverse_dictionary = build_dictionary(read_words(target_train_data_path), vocabulary_size)
    target_train_datas = [sentence_to_onehot(lines, target_dictionary) for lines in read_data(target_train_data_path)]

    source_valid_datas = [sentence_to_onehot(lines, source_dictionary) for lines in read_data(source_valid_data_path)]
    target_valid_datas = [sentence_to_onehot(lines, target_dictionary) for lines in read_data(target_valid_data_path)]

    if args.debug:
      source_train_datas = source_train_datas[:1000]
      target_train_datas = source_train_datas[:1000]
  else:
    with open(dictionary_path['source'], 'rb') as f1, \
         open(dictionary_path['source_reverse'], 'rb') as f2, \
         open(dictionary_path['target'], 'rb') as f3, \
         open(dictionary_path['target_reverse'], 'rb') as f4:
      source_dictionary = pickle.load(f1)
      source_reverse_dictionary = pickle.load(f2)
      target_dictionary = pickle.load(f3)
      target_reverse_dictionary = pickle.load(f4)

  source_test_datas = [sentence_to_onehot(lines, source_dictionary) for lines in read_data(source_test_data_path)]
  target_test_datas = [sentence_to_onehot(lines, target_dictionary) for lines in read_data(target_test_data_path)]

  model = Seq2Seq(vocabulary_size=vocabulary_size,
                  embedding_size=embedding_size,
                  hidden_size=hidden_size,
                  batch_size=batch_size,
                  max_time=max_time,
                  flag_gpu=1)
  model.reset()
  if True: #TODO
    ARR = cuda.cupy
    cuda.get_device(0).use()
    model.to_gpu(0)
  else:
    ARR = np
  
  minibatch_idx = {'train': 0, 'valid': 0, 'test': 0}
  if args.mode == 'train':
    # train
    global_max_step = train_step * (len(source_train_datas) // batch_size + 1)
    loss_freq = global_max_step // 100 if global_max_step > 100 else 1
    loss_log = []
    batch_loss_log = []
    loss_suffix = ''
    es = EarlyStopper(max_size=5, edge_threshold=0.1)
    m = Monitor(global_max_step)
    log = Logger('%s/log' % model_directory)
    global_step = 0
    stop_flag = False
    for batch in range(train_step):
      if stop_flag:
        break
      current_batch_loss_log = []
      while True: # minibatch process
        m.monitor(global_step, loss_suffix)
        source_train_batch, _ = batchnize(source_train_datas, batch_size, minibatch_idx['train'])
        target_train_batch, minibatch_idx['train'] = batchnize(target_train_datas, batch_size, minibatch_idx['train'])
        batch_data = seq2seq(source_train_batch, target_train_batch, max_time, vocabulary_size, reverse=True)
        # c2. update weight
        opt = optimizers.Adam()
        opt.setup(model)
        opt.add_hook(optimizer.GradientClipping(5))
        total_loss = model.forward(encoder_words=batch_data['encoder_inputs'],
                                   decoder_words=batch_data['decoder_labels'],
                                   model=model,
                                   ARR=ARR)
        total_loss.backward()
        opt.update()
        opt.reallocate_cleared_grads()

        if global_step % loss_freq == 0:
          source_valid_batch, _ = batchnize(source_valid_datas, batch_size, minibatch_idx['valid'])
          target_valid_batch, minibatch_idx['valid'] = batchnize(target_valid_datas, batch_size, minibatch_idx['valid'])
          batch_data = seq2seq(source_valid_batch, target_valid_batch, max_time, vocabulary_size, reverse=True)
          # c3. calc loss
          loss = model.forward(encoder_words=batch_data['encoder_inputs'],
                               decoder_words=batch_data['decoder_labels'],
                               model=model,
                               ARR=ARR)
          loss_data = cuda.to_cpu(loss.data)
          loss_log.append(loss_data)
          current_batch_loss_log.append(loss_data)
          loss_suffix = 'loss: {}'.format(loss_data)
        global_step += 1
        if minibatch_idx['train'] == 0:
          batch_loss = np.mean(current_batch_loss_log)
          batch_loss_log.append(batch_loss)
          loss_msg = 'Batch: {}/{}, batch loss: {}'.format(batch + 1, train_step, batch_loss)
          print(loss_msg)
          log('%s\n' % loss_msg)
          es_status = es(batch_loss)
          if batch > train_step // 2 and es_status:
            print('early stopping at step: %d' % global_step)
            stop_flag = True
          break

    # c4. save tf.graph and variables
    serializers.save_hdf5(model_path, model)
    print('save at %s' % model_path)

    # save plot of loss
    plt.plot(np.arange(len(loss_log)) * loss_freq, loss_log)
    plt.savefig('%s_global_loss.png' % model_path)
    plt.figure()
    plt.plot(np.arange(len(batch_loss_log)), batch_loss_log)
    plt.savefig('%s_batch_loss.png' % model_path)

    # save dictionary
    with open(dictionary_path['source'], 'wb') as f1, \
         open(dictionary_path['source_reverse'], 'wb') as f2, \
         open(dictionary_path['target'], 'wb') as f3, \
         open(dictionary_path['target_reverse'], 'wb') as f4:
      pickle.dump(source_dictionary, f1)
      pickle.dump(source_reverse_dictionary, f2)
      pickle.dump(target_dictionary, f3)
      pickle.dump(target_reverse_dictionary, f4)

  elif args.mode == 'eval':
    # c5.
    serializers.load_hdf5(model_path, model)
    print('load from %s' % model_path)

  else:
    raise # args.mode should be train or eval

  # evaluate
  #loss_val = [] # abandoned
  input_vectors = None
  predict_vectors = None
  for i in range(len(source_test_datas) // batch_size + 1):
    source_test_batch, _ = batchnize(source_test_datas, batch_size, minibatch_idx['test'])
    target_test_batch, minibatch_idx['test'] = batchnize(target_test_datas, batch_size, minibatch_idx['test'])
    batch_data = seq2seq(source_test_batch, target_test_batch, max_time, vocabulary_size, reverse=True)
    # c6.
    """
    pred = model.decode(batch_data[i])
    if predict_vectors is None:
      predict_vectors = pred.T
    else:
      predict_vectors = np.vstack((predict_vectors, pred.T))
    """
    input_ = batch_data['encoder_inputs']
    if input_vectors is None:
      input_vectors = input_.T
    else:
      input_vectors = np.vstack((input_vectors, input_.T))
    pred = model.forward_test(encoder_words=batch_data['encoder_inputs'],
                              model=model,
                              ARR=ARR)
    predict_vectors = pred.T.tolist()

  input_sentences = ''
  predict_sentences = ''
  ignore_token = EOS
  for i, (input_vector, predict_vector) in enumerate(zip(input_vectors[:len(source_test_datas)], predict_vectors[:len(target_test_datas)])):
    input_sentences += ' '.join([source_reverse_dictionary[vector] for vector in input_vector if not vector == ignore_token])
    predict_sentences += ' '.join([target_reverse_dictionary[vector] for vector in predict_vector if not vector == ignore_token])
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
  #print('mean of loss: %f' % np.mean(loss_val))

  print('finish.')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', '-m', type=str, help='train | eval')
  parser.add_argument('--config', '-c', type=str, help='config file path')
  parser.add_argument('--debug', '-d', type=bool, default=False, help='flag of debug mode')
  parser.add_argument('--overwrite', '-ow', type=bool, default=False, help='flag of forcely overwrite')
  args = parser.parse_args()
  main(args)
  
