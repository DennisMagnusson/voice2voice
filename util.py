import json

from scipy.io import wavfile
from scipy import signal

import torch

from UniversalVocoding.preprocess import get_mel
from UniversalVocoding.model import Vocoder

import soundfile

import torch

def wav_to_mel(filename):
  #sample_rate, samples = wavfile.read(filename)
  #freq, times, spectrogram = signal.spectrogram(samples, sample_rate)

  return get_mel(filename)

def get_vocoder(device):
  with open('UniversalVocoding/config.json') as f:
    params = json.load(f)


  model = Vocoder(mel_channels=params["preprocessing"]["num_mels"],
                    conditioning_channels=params["vocoder"]["conditioning_channels"],
                    embedding_dim=params["vocoder"]["embedding_dim"],
                    rnn_channels=params["vocoder"]["rnn_channels"],
                    fc_channels=params["vocoder"]["fc_channels"],
                    bits=params["preprocessing"]["bits"],
                    hop_length=params["preprocessing"]["hop_length"])
  model.to('cpu')

  state_dict = torch.load('pretrained_models/model.ckpt-100000.pt', map_location=torch.device('cpu'))

  model.load_state_dict(state_dict['model'])
  return model

def mel_to_wav(vocoder, mel):
  return vocoder.generate(torch.FloatTensor(mel).unsqueeze(0))

def write_wav(wav, filename):
  sample_rate = 16000
  soundfile.write(filename, wav, sample_rate)
  



