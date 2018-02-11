# coding: utf-8
#
# 入力したシーケンスをそのまま返すseq2seqモデルを学習する。
#

# # おまじない

# In[1]:

import tensorflow as tf


# In[2]:

tf.reset_default_graph()
sess = tf.InteractiveSession()


# # データ入出力

# In[3]:

import numpy as np


# In[4]:

PAD = 0
EOS = 1


# 1センテンスに対応するdataという単位の変数を作成する。

# In[5]:

def make_data(max_time=5, vocabulary_size=10):
  time = np.random.randint(1, max_time)
  return np.random.randint(2, vocabulary_size, size=(time))


# In[6]:

data = make_data()


# In[7]:

data


# 複数のデータからひとつのミニバッチを作成する。
# 
# * create_batch: 1からバッチを作成, 第一引数にdataのリストがあれば、それを利用する

# In[8]:

def padding(vector, max_size):
  new_v = []
  for i in range(max_size):
    item = 0 if len(vector) <= i else vector[i]
    new_v.append(item)
  return new_v


# In[9]:

def create_batch(datas=[], max_time=5, batch_size=8, vocabulary_size=10):
  # make datas
  if datas == []:
    for _ in range(batch_size):
      data = make_data(max_time, vocabulary_size)
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


# In[10]:

datas = []
max_time = 5
batch_size = 8
vocabulary_size = 10


# In[11]:

create_batch(datas, max_time, batch_size, vocabulary_size)


# ## placeholder

# In[12]:

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')


# In[13]:

decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')


# In[14]:

decoder_labels = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_labels')


# # 埋め込み

# In[15]:

input_embedding_size = 20


# ## variable and formula

# In[16]:

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)


# In[17]:

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)


# In[18]:

decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)


# # エンコーダ

# In[19]:

encoder_units = 20


# ## formula

# In[20]:

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_units)


# In[21]:

_, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
    dtype=tf.float32, time_major=True
)


# In[22]:

encoder_final_state


# # デコーダ

# In[23]:

decoder_units = encoder_units


# ## formula

# In[24]:

decoder_cell = tf.contrib.rnn.LSTMCell(decoder_units)


# In[25]:

decoder_output, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, decoder_inputs_embedded,
    initial_state=encoder_final_state,
    scope="plain_decoder",
    dtype=tf.float32, time_major=True
)


# In[26]:

decoder_logits = tf.contrib.layers.linear(decoder_output, vocabulary_size)


# In[27]:

decoder_prediction = tf.argmax(decoder_logits, 2) # max_time: axis=0, batch: axis=1, vocab: axis=2


# # オプティマイザ

# ## formula

# In[28]:

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_labels, depth=vocabulary_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)


# # 初期化

# In[29]:

sess.run(tf.global_variables_initializer())


# # 訓練

# In[30]:

train_step = 5000


# In[31]:

for i in range(train_step):
  batch_data = create_batch(datas, max_time, batch_size, vocabulary_size)
  sess.run(fetches=[train_op, loss],
           feed_dict={encoder_inputs:batch_data['encoder_inputs'],
                      decoder_inputs:batch_data['decoder_inputs'],
                      decoder_labels:batch_data['decoder_labels']})
  if i % 1000 == 0:
    loss_val = sess.run(fetches=loss,
                        feed_dict={encoder_inputs:batch_data['encoder_inputs'],
                                   decoder_inputs:batch_data['decoder_inputs'],
                                   decoder_labels:batch_data['decoder_labels']})
    print('loss: %f' % loss_val)


# # 確認

# In[32]:

pred = sess.run(fetches=decoder_prediction,
                feed_dict={encoder_inputs:batch_data['encoder_inputs'],
                           decoder_inputs:batch_data['decoder_inputs'],
                           decoder_labels:batch_data['decoder_labels']})


# In[33]:

input_ = batch_data['encoder_inputs']


print('input sequences...\n{}'.format(input_))
print('predict sequences...\n{}'.format(pred))
