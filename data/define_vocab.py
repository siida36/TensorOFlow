import argparse

def main(args):
  with open(args.input, 'r') as f:
    lines = f.readlines()

  word_to_count = dict()
  for line in lines:
    words = line.strip().split()
    for word in words:
      if word in word_to_count.keys():
        word_to_count[word] += 1
      else:
        word_to_count[word] = 1

  sorted_w2c = sorted(word_to_count.items(), key=lambda x: x[1], reverse=True)
  vocabulary = []
  for n_val, (key, val) in enumerate(sorted_w2c): 
    if n_val < args.vocabulary_size:
      vocabulary.append(key)
    else:
      break

  with open(args.output, 'w') as f:
    f.write('\n'.join(vocabulary))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', '-i', type=str, help='Input sentence file.', required=True)
  parser.add_argument('--output', '-o', type=str, help='Output vocabulary file name.', required=True)
  parser.add_argument('--vocabulary_size', '-v', type=int, help='Vocabulary size.', required=True)
  args = parser.parse_args()
  main(args)
