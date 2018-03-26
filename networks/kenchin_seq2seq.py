import numpy as np
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
from typing import List, Sequence, TypeVar

class LSTM_Encoder(Chain):
  def __init__(self, vocab_size: int, embed_size: int, hidden_size: int) -> None:
    """
    Parameters:
      vocabulary_size: 
      embedding_size: 
      hidden_size: hidden units of Encoder and Decoder
    """
    super(LSTM_Encoder, self).__init__(
      # onehot to embed
      xe = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
      # 4 times embed (for LSTM-linear, input-gate, output-gate, forget-gate)
      eh = links.Linear(embed_size, 4 * hidden_size),
      # 4 times output of Linear as LSTM
      hh = links.Linear(hidden_size, 4 * hidden_size)
    )

  def __call__(self, x, c, h):
    """
    Parameters:
      x: one-hot
      c: cell state
      h: hidden units

    Return:
      Two Variable objects (c, h)
      c: cell state
      h: outgoing signal
    """
    # xeで単語ベクトルに変換して、そのベクトルをtanhにかける
    e = functions.tanh(self.xe(x))
    # 前の内部メモリの値と単語ベクトルの4倍サイズ、中間層の4倍サイズを足し合わせて入力
    return functions.lstm(c, self.eh(e) + self.hh(h))


class LSTM_Decoder(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size):
    """
    クラスの初期化
    :param vocab_size: 使われる単語の種類数（語彙数）
    :param embed_size: 単語をベクトル表現した際のサイズ
    :param hidden_size: 中間ベクトルのサイズ
    """
    super(LSTM_Decoder, self).__init__(
      # 入力された単語を単語ベクトルに変換する層
      ye = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
      # 単語ベクトルを中間ベクトルの4倍のサイズのベクトルに変換する層
      eh = links.Linear(embed_size, 4 * hidden_size),
      # 中間ベクトルを中間ベクトルの4倍のサイズのベクトルに変換する層
      hh = links.Linear(hidden_size, 4 * hidden_size),
      # 出力されたベクトルを単語ベクトルのサイズに変換する層
      he = links.Linear(hidden_size, embed_size),
      # 単語ベクトルを語彙サイズのベクトル（one-hotなベクトル）に変換する層
      ey = links.Linear(embed_size, vocab_size)
    )

  def __call__(self, y, c, h):
    """

    :param y: one-hotなベクトル
    :param c: 内部メモリ
    :param h: 中間ベクトル
    :return: 予測単語、次の内部メモリ、次の中間ベクトル
    """
    # 入力された単語を単語ベクトルに変換し、tanhにかける
    e = functions.tanh(self.ye(y))
    # 内部メモリ、単語ベクトルの4倍+中間ベクトルの4倍をLSTMにかける
    c, h = functions.lstm(c, self.eh(e) + self.hh(h))
    # 出力された中間ベクトルを単語ベクトルに、単語ベクトルを語彙サイズの出力ベクトルに変換
    t = self.ey(functions.tanh(self.he(h)))
    return t, c, h


class Seq2Seq(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size, batch_size, flag_gpu=True):
    """
    Seq2Seqの初期化
    :param vocab_size: 語彙サイズ
    :param embed_size: 単語ベクトルのサイズ
    :param hidden_size: 中間ベクトルのサイズ
    :param batch_size: ミニバッチのサイズ
    :param flag_gpu: GPUを使うかどうか
    """
    super(Seq2Seq, self).__init__(
      # Encoderのインスタンス化
      encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
      # Decoderのインスタンス化
      decoder = LSTM_Decoder(vocab_size, embed_size, hidden_size)
    )
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    # GPUで計算する場合はcupyをCPUで計算する場合はnumpyを使う
    if flag_gpu:
      self.ARR = cuda.cupy
    else:
      self.ARR = np

  def encode(self, words):
    """
    Encoderを計算する部分
    :param words: 単語が記録されたリスト
    :return:
    """
    # 内部メモリ、中間ベクトルの初期化
    c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
    h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

    # エンコーダーに単語を順番に読み込ませる
    for w in words:
      c, h = self.encoder(w, c, h)

    # 計算した中間ベクトルをデコーダーに引き継ぐためにインスタンス変数にする
    self.h = h
    # 内部メモリは引き継がないので、初期化
    self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

  def decode(self, w):
    """
    デコーダーを計算する部分
    :param w: 単語
    :return: 単語数サイズのベクトルを出力する
    """
    t, self.c, self.h = self.decoder(w, self.c, self.h)
    return t

  def reset(self):
    """
    中間ベクトル、内部メモリ、勾配の初期化
    :return:
    """
    self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
    self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

    self.zerograds()


def forward(enc_words, dec_words, model, ARR):
  """
  順伝播の計算を行う関数
  :param enc_words: 発話文の単語を記録したリスト
  :param dec_words: 応答文の単語を記録したリスト
  :param model: Seq2Seqのインスタンス
  :param ARR: cuda.cupyかnumpyか
  :return: 計算した損失の合計
  """
  # バッチサイズを記録
  batch_size = len(enc_words[0])
  # model内に保存されている勾配をリセット
  model.reset()
  # 発話リスト内の単語を、chainerの型であるVariable型に変更
  enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
  # エンコードの計算 ⑴
  model.encode(enc_words)
  # 損失の初期化
  loss = Variable(ARR.zeros((), dtype='float32'))
  # <eos>をデコーダーに読み込ませる (2)
  t = Variable(ARR.array([0 for _ in range(batch_size)], dtype='int32'))
  # デコーダーの計算
  for w in dec_words:
    # 1単語ずつデコードする (3)
    y = model.decode(t)
    # 正解単語をVariable型に変換
    t = Variable(ARR.array(w, dtype='int32'))
    # 正解単語と予測単語を照らし合わせて損失を計算 (4)
    loss += functions.softmax_cross_entropy(y, t)
  return loss
