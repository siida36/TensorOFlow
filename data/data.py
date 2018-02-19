import collections
import os

from typing import List, Sequence, TypeVar
import numpy as np

from configs.configs import Configs 

ROOT = os.environ['TENSOROFLOW']
config_file = '%s/configs/const.ini' % ROOT
print('data.py: config file is %s' % config_file)
c = Configs(config_file)
PAD = c.const['PAD']
EOS = c.const['EOS']
BOS = c.const['BOS']
UNK = c.const['UNK']
END_TOKEN = c.const['END_TOKEN']
A = TypeVar('A')

def read_words(input_file: str) -> List[str]:
  words = []
  with open(input_file) as f:
    words += f.read().split()
  return words

def read_data(input_file: str) -> List[List[str]]:
  with open(input_file) as f:
    lines = f.readlines()  
  return lines

def sentence_to_onehot(sentence: str, dictionary: dict) -> List[int]:
  onehots = []
  for word in sentence.strip().split():
    onehot = dictionary[word] if word in dictionary.keys() else UNK
    onehots.append(onehot)
  return onehots

def build_dictionary(words, vocabulary_size):
  """Process raw inputs into a dataset."""
  count = [['PAD', -1], ['EOS', -1], ['BOS', -1], ['UNK', -1]] # reserved
  count.extend(collections.Counter(words).most_common(vocabulary_size - len(count)))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[UNK][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return dictionary, reversed_dictionary

def simple_data(max_time: int, vocabulary_size: int) -> Sequence[int]:
  """
  Examples:
    data = simple_data(max_time=5, vocabulary_size=10)
  """
  #np.random.seed(0) # for debug
  time = np.random.randint(1, max_time)
  return np.random.randint(END_TOKEN, vocabulary_size, size=(time))

def padding(vector: Sequence[A], max_size: int, pad=PAD) -> Sequence[A]:
  if len(vector) >= max_size:
    return vector[:max_size]
  new_v = []
  for i in range(max_size):
    item = pad if len(vector) <= i else vector[i]
    new_v.append(item)
  return new_v

def through(datas: List[List[int]], max_time: int, batch_size: int, vocabulary_size: int) -> dict:
  """
  Examples:
    through_batch = through(datas=[], max_time=5, batch_size=8, vocabulary_size=10)
  """
  # make datas
  if datas == []:
    for _ in range(batch_size):
      data = simple_data(max_time, vocabulary_size)
      datas.append(data)
  # convert to I/O of seq2seq
  decoder_max_time = max_time + 1
  encoder_inputs = [padding(data, max_time) for data in datas]
  decoder_inputs = [padding(np.concatenate([[EOS], data]), decoder_max_time) for data in datas]
  decoder_labels = [padding(np.concatenate([data, [EOS]]), decoder_max_time) for data in datas]
  res = {'encoder_inputs': np.array(encoder_inputs).T, 
         'decoder_inputs': np.array(decoder_inputs).T, 
         'decoder_labels': np.array(decoder_labels).T}
  return res

def batchnize(data: Sequence[A], batch_size: int, batch_idx: int) -> Sequence[A]:
  if batch_size * (batch_idx + 1) <= len(data):
    return data[batch_size * batch_idx: batch_size * (batch_idx + 1)], batch_idx + 1
  last = data[batch_size * batch_idx:]
  over = batch_size - len(last)
  return np.concatenate((last, data[:over])), 0

def seq2seq(source_datas: List[List[int]], target_datas: List[List[int]], max_time: int, vocabulary_size: int, use_BOS=True, decoder_time_append=False, reverse=False) -> dict:
  """
  Examples:
  """
  decoder_max_time = max_time + 1 if decoder_time_append else max_time

  if not use_BOS:
    encoder_inputs = [padding(data, max_time) for data in source_datas]
    decoder_inputs = [padding(np.concatenate([[EOS], data]), decoder_max_time) for data in target_datas]
    decoder_labels = [padding(np.concatenate([data, [EOS]]), decoder_max_time) for data in target_datas]
  else:
    encoder_inputs = [padding(np.concatenate([data, [EOS]]), max_time, pad=EOS) for data in source_datas]
    decoder_inputs = [padding(np.concatenate([[BOS], data, [EOS]]), decoder_max_time, pad=EOS) for data in target_datas]
    decoder_labels = [padding(np.concatenate([data, [EOS]]), decoder_max_time, pad=EOS) for data in target_datas]

  if reverse:
    encoder_inputs = np.fliplr(encoder_inputs)

  res = {'encoder_inputs': np.array(encoder_inputs).T, 
         'decoder_inputs': np.array(decoder_inputs).T, 
         'decoder_labels': np.array(decoder_labels).T}
  return res
