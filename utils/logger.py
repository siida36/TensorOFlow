import os

class Logger(object):

  def __init__(self, log_file, initialize=True):
    self.log_file = log_file
    if initialize:
      if os.path.isfile(log_file):
        os.remove(log_file)

  def __call__(self, data):
    with open(self.log_file, 'a') as f:
      f.write(data)
