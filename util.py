import json

from scipy.io import wavfile
from scipy import signal

import torch

from UniversalVocoding.preprocess import get_mel
from UniversalVocoding.model import Vocoder

import soundfile

import torch

def wav_to_mel(filename, config_filename='UniversalVocoding/config.json'):
  #sample_rate, samples = wavfile.read(filename)
  #freq, times, spectrogram = signal.spectrogram(samples, sample_rate)
  return get_mel(filename, config_filename=config_filename)

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


if __name__=='__main__':
  from os import listdir
  import librosa
  import librosa.display
  from matplotlib import pyplot as plt
  base_folder = './speech_data/16khz/'
  i = 0
  filenames = [base_folder+x for x in listdir(base_folder) if x.endswith('.wav')]
  for filename in filenames:
    print(filename)
    mel1 = wav_to_mel(filename, 'UniversalVocoding/config.json')
    mel2 = wav_to_mel(filename, 'UniversalVocoding/config2.json')

    s1 = mel1.shape[0]
    s2 = mel2.shape[0]

    #print('len1 = ' + str(s1))
    #print('len2 = ' + str(s2))
    #print(s1*2.5-s2)
    i += 1
    """
    librosa.display.specshow(mel1.T, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('mel1')
    plt.show()
    librosa.display.specshow(mel2.T, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('mel2')
    plt.show()
    """
    #print(mel1.shape[0])
    #print(mel2.shape)
    if i == 50:
      die()

  
  


