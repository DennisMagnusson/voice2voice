import numpy as np
import torch

from tqdm import tqdm
from phoneme_classifier.model import PhonemeClassifier
from phoneme_classifier.read_data import get_wav, wav_to_mel, get_pyin

from os import listdir

def preprocess(phoneme_classifier, vocoder, output_dir, seq_len=256, fmax=500, k=4):
  base_folder = './speech_data/LJSpeech-1.1/wavs/'
  filenames = [x for x in listdir(base_folder) if x.endswith('.wav')]

  for filename in tqdm(filenames):
    #mel = get_mel(base_folder+filename, config_filename='./UniversalVocoding/config2.json')
    wav, sr = get_wav(base_folder+filename)
    mel_phoneme = wav_to_mel(wav, sr).T
    mel_vocoder = vocoder(torch.Tensor([wav]))[0].T

    #f0, prob = get_pyin(wav, sr)
    f0, prob = get_pyin(wav, sr, k=k, fmax=fmax, probs=True)
    with torch.no_grad():
        # Apply softmax with 0.7 temperature (the best temperature)
        #batch_y, _ = F.softmax(phoneme_classifier(batch_x)/0.7)
        batch_x, _ = phoneme_classifier(torch.Tensor(mel_phoneme).unsqueeze(0))

    f0 = torch.Tensor(f0).unsqueeze(0).unsqueeze(2)
    prob = torch.Tensor(prob).unsqueeze(0).unsqueeze(2)
    inpu = torch.cat((batch_x.transpose(1, 2), f0, prob), 2)

    mel_vocoder = (mel_vocoder.numpy() + 5.0) / 2.0

    np.save(output_dir + filename + 'x.npy', inpu)
    np.save(output_dir + filename + 'y.npy', mel_vocoder)

  
def get_vocoder():
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    return vocoder

if __name__ == '__main__':
  phoneme_classifier = PhonemeClassifier()
  phoneme_classifier.load_state_dict(torch.load('./phoneme_classifier/models/phoneme-classifier-final64-state_dict-30ep.pth'))
  phoneme_classifier.eval()

  vocoder = get_vocoder()

  preprocess(phoneme_classifier, vocoder, './processed_data_with_probs/', seq_len=256, fmax=500, k=1)

