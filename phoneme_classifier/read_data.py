import librosa
import numpy as np
import scipy
from scipy.io import wavfile

import sys
sys.path.append('..')
#from .. import wav_to_mel 
from util import wav_to_mel

def get_data():
  
  pass


def read_data(path):
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
  mel = wav_to_mel(wav_filename, './UniversalVocoding/config2.json')


  vocab_dict = get_dict()

  phones = np.zeros((len(mel), (len(vocab_dict))))
  print(phones.shape)
  for i in phn_file:
    start, end, label = i.split()
    start = int(start) // 80
    end = int(end) // 80
    #phones[start-start_index:end-start_index][vocab_dict[label]] = 1
    phones[start:end, vocab_dict[label]] = 1
  
  return mel, phones


def get_dict():
  vocab = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

  vocab_dict = {vocab[i] : i for i in range(len(vocab))}

  return vocab_dict

