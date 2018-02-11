import configparser
import os
import re

class Configs(object):

  def __init__(self, config_file):
    """
    How to add sections and attributes:
    1. define member, i.e. self.option = dict()
    2. define section_pairs, i.e. section_pairs...
    3. write configs in your config file. i.e.
      [option]
      train_step : 5000

    Common config files are unique files in package.
    If you use common config files:
    1. define member, i.e. self.const = dict()
    2. define common_section_pairs, i.e. common_section_pairs...
    3. write configs in your config file. i.e.
       [const]
       BEGIN_TOKEN : 0
    4. write section and filepath in all config files. i.e.
       [common]
       const : configs/const.ini
    """
    ROOT = os.environ['TF_TOUJIKA']
    self.option = dict()
    self.data = dict()
    self.const = dict()
    
    sections, attributes = self.get_attr(config_file)
    section_pairs = {'option': self.option,
                     'data': self.data}
    for k, v in zip(section_pairs.keys(), section_pairs.values()):
      if k in sections:
        self.read_config(config_file, v, k, attributes[k])

    # common config file
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file, 'UTF-8')
    
    common_section_pairs = {'const': self.const}
    for k, v in zip(common_section_pairs.keys(), common_section_pairs.values()):
      common_file = '{}/{}'.format(ROOT, config_parser.get('common', k))
      common_sections, common_attributes = self.get_attr(common_file)
      self.read_config(common_file, v, k, common_attributes[k])

  def read_config(self, config_file, section, section_name, attr):
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file, 'UTF-8')
    for a in attr:
      buf = config_parser.get(section_name, a)
      section[a] = buf if not str.isdigit(buf) else int(buf)

  def get_attr(self, config_file):
    with open(config_file, 'r') as f:
      config_lines = f.readlines()
    sections = []
    current_section = ''
    attributes = dict()
    for n, config_line in enumerate(config_lines):
      config_line = config_line.strip()
      if not re.match('^\[.*]$', config_line) is None:
        section = config_line.strip('[').strip(']')
        sections.append(section)
        current_section = section
      else:
        attribute = config_line.split(':')[0].strip()
        if attribute == '':
          continue
        if not current_section in attributes.keys():
          attributes[current_section] = list()
        attributes[current_section].append(attribute)
    return sections, attributes


  def old_init(self, config_file):
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file, 'UTF-8')
    const_attr = ['BEGIN_TOKEN',
                  'END_TOKEN',
                  'PAD',
                  'EOS',
                  'BOS',
                  'UNK']
    const = dict()
    for a in const_attr:
      buf = config_parser.get('const', a)
      const[a] = buf if not str.isdigit(buf) else int(buf)
    self.const = const

    option_attr = ['train_step',
                   'max_time',
                   'batch_size',
                   'vocabulary_size',
                   'embedding_size',
                   'hidden_units']
    option = dict()
    for a in option_attr:
      buf = config_parser.get('option', a)
      option[a] = buf if not str.isdigit(buf) else int(buf)
    self.option = option



