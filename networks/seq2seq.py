from typing import List, Sequence, TypeVar

from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import numpy as np


class LSTM_Encoder(Chain):
  def __init__(self, vocabulary_size: int, embedding_size: int, hidden_size: int) -> None:
    """
    Parameters:
      vocabulary_size: 
      embedding_size: 
      hidden_size: hidden units of Encoder and Decoder
    """
    super(LSTM_Encoder, self).__init__(
      # onehot to embed
      xe = links.EmbedID(vocabulary_size, embedding_size, ignore_label=-1),
      # 4 times embed (for LSTM-linear, input-gate, output-gate, forget-gate)
      eh = links.Linear(embedding_size, 4 * hidden_size),
      # 4 times output of Linear as LSTM
      hh = links.Linear(hidden_size, 4 * hidden_size)
    )

  def __call__(self, x: Sequence, c: Sequence, h: Sequence) -> (Sequence, Sequence):
    """
    Parameters:
      x: one-hot
      c: cell state
      h: hidden units

    Return:
      two Variable objects (c, h)
      c: cell state
      h: outgoing signal
    """
    # word to embed, and then activate
    e = functions.tanh(self.xe(x))
    # compute lstm cell by inputting internal memory and 4-timed embed + 4-timed output of lstm
    return functions.lstm(c, self.eh(e) + self.hh(h))


class LSTM_Decoder(Chain):
  def __init__(self, vocabulary_size, embedding_size, hidden_size):
    """
    Parameters:
      vocabulary_size: 
      embedding_size: 
      hidden_size:
    """
    super(LSTM_Decoder, self).__init__(
      # onehot to embed
      ye = links.EmbedID(vocabulary_size, embedding_size, ignore_label=-1),
      # 4 times embed 
      eh = links.Linear(embedding_size, 4 * hidden_size),
      # 4 times output of Linear
      hh = links.Linear(hidden_size, 4 * hidden_size),
      # output to embed
      he = links.Linear(hidden_size, embedding_size),
      # embed to onehot
      ey = links.Linear(embedding_size, vocabulary_size)
    )

  def __call__(self, y: Sequence, c: Sequence, h: Sequence) -> (List[int], Sequence, Sequence):
    """
    Parameters:
      y: one-hot
      c: cell state
      h: outgoing memory
    Returns:
      predicted word and two Variable objects (t, c, h)
      t: predicted word
      c: cell state
      h: outgoing signal
    """
    # onehot to embed, and then activate
    e = functions.tanh(self.ye(y))
    # compute lstm cell by inputting internal memory and 4-timed embed + 4-timed output of lstm
    c, h = functions.lstm(c, self.eh(e) + self.hh(h))
    # output of lstm to embed, and then embed to onehot 
    t = self.ey(functions.tanh(self.he(h)))
    return t, c, h


class Seq2Seq(Chain):
  def __init__(self, vocabulary_size: int, embedding_size: int, hidden_size: int, batch_size: int, max_time: int, flag_gpu=True) -> None:
    """
    Parameters:
      vocabulary_size:
      embedding_size:
      hidden_size:
      batch_size:
      max_time:
      flag_gpu:
    """
    super(Seq2Seq, self).__init__(
      encoder = LSTM_Encoder(vocabulary_size, embedding_size, hidden_size),
      decoder = LSTM_Decoder(vocabulary_size, embedding_size, hidden_size)
    )
    self.vocabulary_size = vocabulary_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.max_time = max_time

    if flag_gpu:
      self.ARR = cuda.cupy
    else:
      self.ARR = np

  def encode(self, words: List[List[int]]) -> None:
    """
    Paramater:
      words: list of sentences(as list of word), dim is (batch, max_time)
    """
    # initial cell state and output of lstm
    c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
    h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

    # input word to encoder
    for w in words:
      c, h = self.encoder(w, c, h)

    # for inheriting cell state
    self.h = h

    # initialize output of lstm (no need)
    self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

  def decode(self, w: List[List[int]]) -> Sequence:
    """
    Parameters:
      words: list of sentences(as list of word), dim is (batch, max_time)
    Returns:
      output of decoder
      t: onehot
    """
    t, self.c, self.h = self.decoder(w, self.c, self.h)
    return t

  def reset(self) -> None:
    """
    reset output of lstm, cell state, and gradients
    :return:
    """
    self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
    self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

    self.zerograds()


  def forward(self, encoder_words: List[List[int]], decoder_words: List[List[int]], model: object, ARR: object):
    """
    Parameters:
      encoder_words: list of sequence(speech)
      decoder_words: list of sequence(responce)
      model: instance of Seq2Seq
      ARR: cuda.cupy or numpy
    Return:
      total loss
    """
    # define batch size by 1st dim of encoder_words
    batch_size = len(encoder_words[0])
    # reset gradient
    model.reset()
    # cast list to Variable
    encoder_words = [Variable(ARR.array(row, dtype='int32')) for row in encoder_words]
    # forward propagation at encoder
    model.encode(encoder_words)
    # reset loss
    loss = Variable(ARR.zeros((), dtype='float32'))
    
    # input <eos> to decoder
    t = Variable(ARR.array([0 for _ in range(batch_size)], dtype='int32'))

    # decode
    for w in decoder_words:
      # input a word
      y = model.decode(t)
      # cast array to Variable
      t = Variable(ARR.array(w, dtype='int32'))
      # calc loss
      loss += functions.softmax_cross_entropy(y, t)
    return loss

  def forward_test(self, encoder_words: List[List[int]], model: object, ARR: object) -> List[List[int]]:
    """
    Parameters:
      encoder_words: list of sequence(speech)
      model: instance of Seq2Seq
      ARR: cuda.cupy or numpy
    Return:
      time-major predected sentences
      sentences: first_dim is sentence, second_dim is minibatch
    """
    model.reset()
    encoder_words = [Variable(ARR.array(row, dtype='int32')) for row in encoder_words]
    model.encode(encoder_words)
    #t = Variable(ARR.array([0], dtype='int32')) # TODO: batch数文のzeroが必要
    #t = Variable(ARR.zeros(shape=(1, self.batch_size), dtype=ARR.int32))
    t = Variable(ARR.array([0 for _ in range(self.batch_size)], dtype='int32'))
    counter = 0
    while counter < self.max_time:
      y = model.decode(t)
      pred_sentences = y.data.argmax(axis=1) #pred_sentences: array([1, 1, 1, ..., n]) (n is batch) 
      if counter > 0:
        sentences = ARR.vstack([sentences, pred_sentences])
      else:
        sentences = pred_sentences
      t = Variable(ARR.array(pred_sentences, dtype='int32'))
      counter += 1
    return sentences
