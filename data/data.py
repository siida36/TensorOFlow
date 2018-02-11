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
END_TOKEN = c.const['END_TOKEN']
A = TypeVar('A')

def simple_data(max_time: int, vocabulary_size: int) -> Sequence[int]:
  """
  Examples:
    data = simple_data(max_time=5, vocabulary_size=10)
  """
  #np.random.seed(0) # for debug
  time = np.random.randint(1, max_time)
  return np.random.randint(END_TOKEN, vocabulary_size, size=(time))

def padding(vector: Sequence[A], max_size: int) -> Sequence[A]:
  new_v = []
  for i in range(max_size):
    item = 0 if len(vector) <= i else vector[i]
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
