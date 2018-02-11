from bisect import bisect
import numpy as np
from typing import List, Sequence, TypeVar


class EarlyStopper(object):

  def __init__(self, max_size: int, edge_threshold: int, 
               sample=100, lower_threshold=30):
    self.queue = []
    self.edge = []
    self.max_size = max_size # number of queue size
    self.edge_threshold = edge_threshold # stopping threshold of edge
    self.sample = sample # number of sample edge
    self.lower_threshold = lower_threshold # stopping threshold of rank with edge distribution

  def __call__(self, data: float):
    self.enqueue(data)
    return self.check_status()

  def enqueue(self, data: float):
    if len(self.queue) < self.max_size:
      self.queue.append(data)
    else:
      _ = self.dequeue()
      self.enqueue(data)
     
  def dequeue(self) -> float:
    data = self.queue[0]
    self.queue = self.queue[1:]
    return data

  def get_edge(self) -> List[float]:
    edge = []
    if len(self.queue) < 2:
      return edge
    for i in range(0, len(self.queue) - 1):
      edge.append(abs(self.queue[i + 1] - self.queue[i]))
    return edge

  def sample_edge(self, sample=100):
    if len(self.edge) < sample:
      edge = self.get_edge()
      for e in edge:
        if len(self.edge) < sample:
          self.edge.append(e)
        else:
          return False # sampling had finished
      return True
    else:
      return False # sampling had finished
    
  def check_status(self):
    edge = self.get_edge()
    if edge == []:
      return False
    recent_edge = np.mean(self.get_edge())
    if recent_edge < self.edge_threshold:
      return True
    else:
      return False

  def check_status_with_sample(self):
    """ TODO: statistical sampling
    """
    if self.sample_edge(self.sample):
      return False
    rank = sorted(self.edge)
    recent_edge = np.mean(self.get_edge())
    if bisect(rank, recent_edge) < self.lower_threshold:
      return True
    else:
      return False

