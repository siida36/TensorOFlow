# coding: utf-8
import argparse
import MeCab
import re
import subprocess
import sys


def convert_to_utf8(src_file, dst_file):
  """ Aozora-bunko distribution file is shift-jis, therefore convert to utf8
  """
  subprocess.call('nkf -w {} > {}'.format(src_file, dst_file), shell=True)

def main(args):
  src_file = args.input
  src_file_utf8 = src_file + '.utf8'
  convert_to_utf8(src_file, src_file_utf8)
  with open(src_file_utf8, 'r') as f:
    lines = list(f)

  EOS = tuple('。')
  conditional_EOS = tuple('、')
  talk_begin = ('「', '（')
  talk_end = ('」', '）')
  remove_begin = ('《','［')
  remove_end = ('》', '］')
  stopword = tuple('｜')
  special_char = remove_begin + remove_end + talk_begin + talk_end + stopword
  header_signal = '-' * 30
  footer_signal = '底本：'
  stack = []
  sentences = []
  
  # remove header
  header_counter = 0
  for i, line in enumerate(lines):
    if header_counter >= 2:
      if not re.match('^\n$', line) is None: # count LF
        pass
      else:
        header_line = i
        break
    if header_signal in line:
      header_counter += 1

  # remove footer
  footer_counter = 0
  for i, line in enumerate(reversed(lines)):
    if footer_counter >= 1:
      if not re.match('^\n$', line) is None: # count LF
        pass
      else:
        footer_line = i
        break
    if footer_signal in line:
      footer_counter += 1

  # main process
  # rule 1: if remove_begin is found, begin remove_mode
  # rule 2: if talk_begin is found, begin talk_mode
  # rule 3: if not in remove_mode or talk_mode and EOS is found, enter new-line-process
  talk_mode = False
  remove_mode = False
  mecab = MeCab.Tagger("-Owakati")
  for line in lines[header_line:-(footer_line)]:
    process_flag = False
    for sc in special_char:
      if sc in line:
        process_flag = True
    new_line = ''
    for n_char, char in enumerate(line.strip()):
      if char in talk_begin:
        talk_mode = True
      elif char in remove_begin:
        remove_mode = True
      elif char in talk_end:
        talk_mode = False
      elif char in remove_end:
        remove_mode = False

      # new_line process or adding char
      if (char in EOS or (char in conditional_EOS and n_char == len(line.strip()) - 1) or char in talk_end) and \
         (talk_mode == False and remove_mode == False):
        new_line += char
        new_line = re.sub('^　', '', new_line) # remove head space
        if args.wakachi:
          new_line = mecab.parse(new_line).strip()
        if not new_line.strip() == '':
          sentences.append(new_line)
        new_line = ''
        continue
      elif char in stopword or char in remove_end: 
        pass
      elif remove_mode:
        pass
      elif char in talk_begin and n_char > 0:
        new_line += '\n%s' % char
      else:
        new_line += char

  # output
  dst_file = args.output
  with open(dst_file, 'w') as f:
    f.write('\n'.join(sentences))

  print('output as {}'.format(dst_file))
  subprocess.call(['rm', src_file_utf8])


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Parse text file from Aozora-bunko')
  parser.add_argument('--input', '-i', type=str, help='Input text file.', required=True)
  parser.add_argument('--output', '-o', type=str, help='Output file name.', required=True)
  parser.add_argument('--wakachi', '-w', type=bool, help='Flag of using MeCab', default=False)
  args = parser.parse_args()
  main(args)
