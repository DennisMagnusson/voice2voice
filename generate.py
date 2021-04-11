import librosa
import soundfile

import torch

from phoneme_classifier.model import PhonemeClassifier
from phoneme_classifier.read_data import get_wav, wav_to_mel, get_pyin

from sys import argv
from train import MelGenerator

def generate(inp_filename, outp_filename, phoneme_classifier, mel_generator, vocoder):
  with torch.no_grad():
    wav, sr = get_wav(inp_filename)
    f0 = torch.Tensor(get_pyin(wav, sr, k=1, probs=False)).unsqueeze(0).unsqueeze(2)

    mel = wav_to_mel(wav, sr)
    ph, _ = phoneme_classifier.forward(torch.Tensor(mel.T).unsqueeze(0))
    ph = ph.transpose(1, 2)
    inpu = torch.cat((ph, f0), 2)
    outpu, _ = mel_generator(inpu)
    outpu = (outpu  * 2.0 - 5.0).transpose(1, 2)
    gen_wav = vocoder.inverse(outpu)[0].numpy()
    soundfile.write(outp_filename, gen_wav, 22500)

if __name__ == '__main__':
  vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
  phoneme_classifier = PhonemeClassifier()
  phoneme_classifier.load_state_dict(torch.load('./phoneme_classifier/models/phoneme-classifier-final64-state_dict-30ep.pth'))

  mel_gen = MelGenerator()
  mel_gen.load_state_dict(torch.load('models/15000it-state_dict.pth', map_location=torch.device('cpu')))

  generate(argv[1], argv[2], phoneme_classifier, mel_gen, vocoder)



