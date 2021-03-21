import librosa
from librosa.feature import melspectrogram

import numpy as np
import scipy
from scipy.io import wavfile

import sys
sys.path.append('..')
#from .. import wav_to_mel 
#from util import wav_to_mel

from random import shuffle

from os import walk, listdir

from tqdm import tqdm

import torch 

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

def get_wav(filename):
  # TODO Pad to make both the same length
  wav, sr = librosa.core.load(filename)
  wav = np.pad(wav, 1024+48-(len(wav) % 1024))
  return wav, sr

def get_pyin(wav, sr, k=4, fmax=500):
  frame_length = 256*k
  #f0, _, prob = librosa.pyin(wav, sr=sr, fmin=65, fmax=600, frame_length=frame_length, fill_na=65)
  f0 = librosa.yin(wav, sr=sr, fmin=65, fmax=fmax, frame_length=frame_length)
  f0 = (librosa.hz_to_mel(f0)-librosa.hz_to_mel(65))/librosa.hz_to_mel(fmax)
  f0 = torch.Tensor(f0).repeat_interleave(k)
  #prob = torch.Tensor(prob).repeat_interleave(k)
  #return f0, prob
  return f0


def wav_to_mel(wav, sr):
  mel = melspectrogram(wav, sr=sr, n_fft=256, hop_length=64, win_length=256, n_mels=80, center=True)
  mel = np.log10(np.maximum(mel, 1e-5))
  #mel = melspectrogram(wav, sr=sr, n_fft=2048, hop_length=64, win_length=256, n_mels=80, center=False)

  return mel

def read_data(path):
  global shortest
  wav_filename = path + '.WAV.wav'
  phn_filename = path + '.PHN'

  # Phns
  with open(phn_filename) as f:
    phn_file = f.readlines()

  start_index = int(phn_file[0].split()[1])
  end_index = int(phn_file[-1].split()[0])

  # Mel
  #_, wav = wavfile.read(wav_filename)
  #wav = wav[start_index:end_index]

  #mel = wav_to_mel(wav, './UniversalVocoding/config2.json')
  #mel = wav_to_mel(wav_filename, config_filename='../UniversalVocoding/config2.json')
  #wav, sr = get_wav(wav_filename)
  wav, sr = librosa.core.load(wav_filename, sr=16000)
  #wav = wav[start_index:end_index]
  mel = wav_to_mel(wav, sr).T

  vocab_dict = get_dict()

  #phones = np.zeros((len(mel), (len(vocab_dict))))
  phones = np.zeros((len(mel)), dtype=np.int32)
  #print(phones.shape)
  for i in phn_file:
    start, end, label = i.split()
    start = int(start) // 64
    end = int(end) // 64
    phones[start:end] = vocab_dict[label]

  mel, phones = trim(mel, phones)
  #print('shortest: {}'.format(shortest))
  return mel, phones


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

def get_dict():
  vocab = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

  vocab_dict = {vocab[i] : i for i in range(len(vocab))}

  return vocab_dict

