import librosa
import numpy as np
import scipy

import torch
from scipy.io import wavfile

from UniversalVocoding import wav_to_mel

from random import shuffle

from os import walk, listdir

from tqdm import tqdm

"""
def get_data(base_path):
  x = []
  y = []
  i = 0
  for sub in tqdm(walk(base_path), total=471):
    i += 1
    subdir = sub[0]
    files = [x for x in listdir(subdir) if x.endswith('.PHN')]
    for base_filename in files:
      x_batch, y_batch = read_data(subdir + '/' + base_filename[:-4])# Remove suffix
      x.append(x_batch)
      y.append(y_batch)

  # Shuffle
  c = list(zip(x, y))
  shuffle(c)
  x, y = zip(*c)
  return x, y
"""
def get_vocoder_data(filename):
  mel = wav_to_mel(filename, config_filename='./UniversalVocoding/config2.json')
  wav = librosa.load(filename)[0]
  return mel, wav


def read_wavfile(filename, phoneme_classifier):
  phoneme_mel = wav_to_mel(filename, config_filename='./UniversalVocoding/config2.json')
  with torch.no_grad():
    phones = phoneme_classifier(torch.Tensor([phoneme_mel]))
  gen_mel = torch.Tensor(wav_to_mel(filename, config_filename='./UniversalVocoding/config.json'))

  # TODO Maybe trim here?
  #mel, phones = trim(mel, phones)
  return phones, gen_mel 


def trim(mel, phones):
  vocab_dict = get_dict()
  start = 0
  while phones[start+1] == vocab_dict['h#']:
    start += 1

  end = len(mel)
  while phones[end-1] == vocab_dict['h#']:
    end -= 1

  mel = mel[start:end+1]
  phones = phones[start:end+1]

  return mel, phones

