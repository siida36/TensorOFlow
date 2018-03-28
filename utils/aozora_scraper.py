import argparse
import glob
import pathlib
import re
import requests
import shutil
import subprocess

from bs4 import BeautifulSoup as bs


def li_to_works(li):
  return [str(l_).split('.html')[0].split('/')[-1] for l_ in li]

def work_to_url(work_soup):
  a_list = work_soup.find_all('a')
  for a in a_list:
    if not re.match('.*zip.*', str(a)) is None:
      return a.text

def main(args):
  parser_script = 'aozora_parser.py'
  root_url = 'http://www.aozora.gr.jp'
  author_url = '{}/index_pages/person{}.html'.format(root_url, args.author)
  author_request = requests.get(author_url)
  author_soup = bs(author_request.content, 'html.parser')
  # create output directory
  author = str(author_soup.find_all('td')[5]).replace(', ','_').replace('<td>', '').replace('</td>', '')
  dst = '{}/{}'.format(args.output_dir, author)
  if pathlib.Path(dst).exists():
    shutil.rmtree(dst)
  tmp = '/tmp/aozora_scraper'
  if pathlib.Path(tmp).exists():
    shutil.rmtree(tmp)
  pathlib.Path(dst).mkdir()
  pathlib.Path('%s/original' % dst).mkdir()
  pathlib.Path('%s/parsed' % dst).mkdir()
  pathlib.Path(tmp).mkdir()
  # make list of the work
  works = li_to_works(author_soup.find_all('li'))
  author_id_pad = args.author.zfill(6)
  for work in works:
    work_url = '{}/cards/{}/{}.html'.format(root_url, author_id_pad, work)
    work_request = requests.get(work_url)
    work_soup = bs(work_request.content, 'html.parser')
    zip_url = work_to_url(work_soup)
    subprocess.call('wget {}/cards/{}/files/{}'.format(root_url, author_id_pad, zip_url), shell=True)
    subprocess.call('mv {} {}'.format(zip_url, tmp), shell=True)
    subprocess.call('unzip {}/{} -d {}'.format(tmp, zip_url, tmp), shell=True)
    text_file = glob.glob('{}/*.txt'.format(tmp))[0].split('/')[-1]
    card_id = zip_url.split('_')[0]
    text_file_with_id = '{}_{}.txt'.format(text_file.replace('.txt', ''), card_id)
    subprocess.call('mv {}/{} {}/original/{}'.format(tmp, text_file, dst, text_file_with_id), shell=True)
    subprocess.call('python {} -i {}/original/{} -o {}/parsed/{} -w {}'.format(parser_script, dst, text_file_with_id, dst, text_file_with_id, args.wakachi), shell=True)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='LSTM')
  parser.add_argument('--author', '-a', type=str, help='Target author id.', required=True)
  parser.add_argument('--output_dir', '-o', type=str, help='Output directory.', required=True)
  parser.add_argument('--wakachi', '-w', type=bool, help='Flag of using MeCab', default=False)
  args = parser.parse_args()
  main(args)
