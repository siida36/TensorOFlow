from datetime import datetime
import math


MAX_BAR = 30


class Monitor(object):
  
  def __init__(self, length):
    self.idx = 0
    self.length = length
    self.num = '{}/{}'.format(self.idx, self.length)
    self.bar = '.' * MAX_BAR
    self.divider1 = length // 100 if length >= 100 else 1
    self.divider2 = length // MAX_BAR if length >= MAX_BAR else 1

  def __str__(self):
    return '{} [{}] - TIME: {} {}'.format(self.num, self.bar,
                                       datetime.now().strftime('%H:%M:%S'),
                                       self.suffix)

  def monitor(self, idx, suffix=''):
    flag = 0
    if idx % self.divider1 == 0 or idx % self.divider2 == 0:
      self.idx = idx
      self.update_num()
      flag = 1
    if idx % self.divider2 == 0:
      self.update_bar()
      flag = 1
    if flag == 1:
      self.suffix = suffix
      print(self)
  
  def update_bar(self):
    top = self.bar.find('>')
    if top == -1:
      self.bar = '>' + '.'*(MAX_BAR - 1)
    if top == MAX_BAR-1:
      pass
    else:
      self.bar = '='*(top + 1) + '>' + '.'*(MAX_BAR - (top + 2))

  def update_num(self):
    idx = str(self.idx).rjust(int(math.log10(self.length)) + 1, ' ')
    self.num = '{}/{}'.format(idx, self.length)
    
