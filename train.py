import matplotlib.pyplot as plt

import librosa
import librosa.display

import torch
from torch import nn, optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau

from phoneme_classifier.model import PhonemeClassifier
from phoneme_classifier.read_data import get_wav, wav_to_mel, get_pyin

from os import listdir, mkdir

import soundfile

import time


from tqdm import tqdm

import json
import random

import numpy as np

import wandb


class MelGenerator(nn.Module):
  def __init__(self):
    super(MelGenerator, self).__init__()
    self.enclinear0 = nn.Linear(62, 256)
    self.encdropout0 = nn.Dropout(0.2)
    self.enclinear1 = nn.Linear(256, 128)
    self.encdropout1 = nn.Dropout(0.2)

    #self.linear0 = nn.Linear(256, 128)
    #self.cbhg0 = CBHG(128, 8)

    self.conv0 = ConvThingy(128, 128, 3, nn.ReLU())
    self.conv1 = ConvThingy(128, 256, 3, nn.ReLU())
    self.linear1 = nn.Linear(256, 256)
    self.pool = nn.MaxPool1d(8, stride=4, padding=2)

    self.attn0 = AttentionThingy(256, 512, 256)
    self.attn1 = AttentionThingy(256, 512, 256)
    self.attn2 = AttentionThingy(256, 512, 256)

    self.cbhg1 = CBHG(256, 16)

    self.linear2 = nn.Linear(256, 256)

    self.declinear0 = nn.Linear(512, 256)
    self.declinear1 = nn.Linear(256, 80)

    self.postnet = PostNet(5, 512, 80)


  def forward(self, x, hidden=None):
    out = F.relu(self.encdropout0(self.enclinear0(x)))
    out = F.relu(self.encdropout1(self.enclinear1(out)))

    #out = self.cbhg0(out)
    #out = F.relu(self.linear0(out))

    out = out.transpose(1, 2)
    out = self.conv0(out)
    out = self.conv1(out)
    out = self.pool(out)
    out = out.transpose(1, 2)

    out = F.relu(self.linear1(out))

    out0, h0, h1, h2 = self.attn0(out)
    out1, h0, h1, h2 = self.attn1(out0, h0, h1, h2)
    out2, _, _, _ = self.attn2(out1, h0, h1, h2)

    out = self.cbhg1(out0 + out1 + out2)
    #out = self.cbhg1(out)

    #out = F.relu(self.linear2(out))
    #out = self.cbhg2(out)

    out = F.relu(self.declinear0(out))
    out = self.declinear1(out)
    
    mel_out = 3*torch.sigmoid(out)

    post_out = 3*torch.sigmoid(self.postnet(mel_out))

    return mel_out, post_out

class PostNet(nn.Module):
  def __init__(self, n_convs, hidden_size, mel_size):
    super(PostNet, self).__init__()
    self.convs = []
    self.convs.append(ConvThingy(mel_size, hidden_size, 5, F.tanh))
    for _ in range(n_convs-2):
      self.convs.append(ConvThingy(hidden_size, hidden_size, 5, F.tanh))
    
    self.convs.append(ConvThingy(hidden_size, mel_size, 5, None))

  def forward(self, x):
    out = x.transpose(1, 2)
    for c in self.convs:
      out = c(out)
    out = out.transpose(1, 2)
    return out



class AttentionThingy(nn.Module):
  def __init__(self, in_size, hidden_size, out_size):
    super(AttentionThingy, self).__init__()

    self.dropout0 = nn.Dropout(0.3)
    self.dropout1 = nn.Dropout(0.3)
    self.prenet0 = nn.Linear(in_size, in_size)
    self.prenet1 = nn.Linear(in_size, hidden_size)

    self.attnrnn = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)

    self.decrnn0 = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
    self.decrnn1 = nn.GRU(hidden_size, out_size, 1, batch_first=True)

    #self.pad = nn.ConstantPad1d(((kernel_size-1)//2, kernel_size//2), 0)
    #self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=kernel_size//2)

  def forward(self, x, h0=None, h1=None, h2=None):
    out = F.relu(self.dropout0(self.prenet0(x)))
    out = F.relu(self.dropout1(self.prenet1(out)))

    out, h0 = self.attnrnn(out, hx=h0)
    out0, h1 = self.decrnn0(out, hx=h1)
    out, h2 = self.decrnn1(out0 + out, hx=h2)

    return out, h0, h1, h2


class Conv2DThingy(nn.Module):
  def __init__(self, in_size, out_size, kernel_size, activation):
    super(Conv2DThingy, self).__init__()
    #self.bn = nn.BatchNorm2d(out_size)
    self.activation = activation
    #self.pad = nn.ConstantPad1d(((kernel_size-1)//2, kernel_size//2), 0)
    self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=kernel_size//2)

  def forward(self, x):
    #out = self.pad(x)
    out = self.conv(out)
    #out = self.bn(out)
    if self.activation:
      return self.activation(out)
    return out



class ConvThingy(nn.Module):
  def __init__(self, in_size, out_size, kernel_size, activation):
    super(ConvThingy, self).__init__()
    self.bn = nn.BatchNorm1d(out_size)
    self.activation = activation
    self.pad = nn.ConstantPad1d(((kernel_size-1)//2, kernel_size//2), 0)
    self.conv = nn.Conv1d(in_size, out_size, kernel_size)

  def forward(self, x):
    out = self.pad(x)
    out = self.conv(out)
    out = self.bn(out)
    if self.activation:
      return self.activation(out)
    return out
    
class Highway(nn.Module):
  def __init__(self, size):
    super(Highway, self).__init__()
    self.one = nn.Linear(size, size)
    self.two = nn.Linear(size, size)

  def forward(self, x):
    x0 = F.relu(self.one(x))
    x1 = torch.sigmoid(self.two(x))

    return x0 * x1 + x*(1.0 - x1)

class CBHG(nn.Module):
  def __init__(self, size, k):
    super(CBHG, self).__init__()

    convies = []
    for i in range(1, k+1):
      convies.append(ConvThingy(size, size, i, nn.ReLU()))
    
    self.convies = nn.ModuleList(convies)

    pool_size = 2
    self.poolpad = nn.ConstantPad1d(((pool_size-1)//2, pool_size//2), 0)
    self.pool = nn.MaxPool1d(pool_size, stride=1)

    self.proj1 = ConvThingy(k*size, size, 3, nn.ReLU())
    self.proj2 = ConvThingy(size, size, 3, None)

    highways = []
    for i in range(4):
      highways.append(Highway(size))
    self.highways = nn.ModuleList(highways)
    self.gru = nn.GRU(size, size, 1, dropout=0.1, batch_first=True, bidirectional=True)
    
  def forward(self, x):
    out = x.transpose(1, 2)
    outs = []
    for conv in self.convies:
      outs.append(conv(out))

    out = torch.cat(outs, dim=1)
    out = self.pool(self.poolpad(out))
    out = self.proj1(out)
    out = self.proj2(out)
    
    out = out.transpose(1, 2)

    out += x
    for h in self.highways:
      out = h(out)

    out, _ = self.gru(out)
    return out

def generate(model, phoneme_classifier, vocoder, name='1it', device='cpu', k=1):
  files = [a for a in ['LJ042-0033.wav', 'LJ008-0195.wav', 'LJ050-0174.wav']]
  model.eval()
  phoneme_classifier.eval()
  try:
    mkdir('./generated/{}'.format(name))
  except FileExistsError:
    pass

  for filename in files:
    #mel = get_mel('./speech_data/LJSpeech-1.1/wavs/'+filename, config_filename='./UniversalVocoding/config2.json')
    wav, sr = get_wav('./speech_data/LJSpeech-1.1/wavs/'+filename)
    mel = wav_to_mel(wav, sr).T
    mel_vocoder = vocoder(torch.Tensor([wav]))[0].T
    #f0, prob = get_pyin(wav, sr)
    f0 = torch.Tensor(get_pyin(wav, sr, k=k)).to(device)
    with torch.no_grad():
      phones, _ = phoneme_classifier(torch.Tensor([mel]).to(device))
      phones = phones.to(device)
      #phones = F.softmax(phones/0.7, dim=1)
      #inpu = torch.cat((phones.transpose(1, 2), torch.Tensor(f0), torch.Tensor(prob)), 1)
      inpu = torch.cat((phones.transpose(1, 2), f0.unsqueeze(0).unsqueeze(2)), 2)
      mel2_pre, mel2 = model(inpu)
      #print(mel2.shape)
      
      #for ex in [1.0, 1.25, 1.5, 1.75, 2.0]:
      for ex in [1.0]:
        mel3 = torch.pow(mel2, ex)

        mel3 = mel2 * 2.0 - 5.0
        mel3 = torch.transpose(mel3, 1, 2)
        gen_wav = vocoder.inverse(mel3)[0].to('cpu').numpy()
        print('generating ./generated/{}/{}{}'.format(name, str(ex), filename))
        soundfile.write('./generated/{}/{}{}'.format(name, str(ex), filename), gen_wav, 22500)
        
        plt.imshow(mel3[0].to('cpu').numpy(), aspect='auto', origin='lower')
        plt.colorbar()
        
        wandb.log({filename: plt}),
        plt.close()

        plt.imshow(mel_vocoder.to('cpu').T, aspect='auto', origin='lower')
        plt.colorbar()

        #plt.imshow(mel_vocoder, aspect='auto')
        #fig, ax = plt.subplots()
        #img = librosa.display.specshow(librosa.power_to_db(mel_vocoder, ref=np.max), y_axis='mel', x_axis='time', ax=ax)
        #fig.colorbar(img, ax=ax, format="%+2.f dB")
        #wandb.log({filename + 'ground_truth': wandb.Image(fig)})
        wandb.log({filename + 'ground_truth': plt})

        plt.close()



  model.train()

def get_batches(phoneme_classifier, vocoder, batch_size=64, seq_len=256, fmax=500, k=4, device='cpu'):
  base_folder = './processed_data/'
  filenames = [x.split('.')[0] for x in listdir(base_folder) if x.endswith('.npy')]
  random.shuffle(filenames)
  batch_x = []
  batch_y = []

  batch_f0 = []
  #batch_prob = []

  ti = time.time()

  for filename in filenames:
    #mel = get_mel(base_folder+filename, config_filename='./UniversalVocoding/config2.json')
    x = np.load(base_folder + filename + '.wavx.npy')[0]
    y = np.load(base_folder + filename + '.wavy.npy')
    for i in range(0, len(x)-seq_len*4, seq_len*4):
      batch_x.append(x[i:(i+seq_len*4)])
      batch_y.append(y[i//4:(i//4+seq_len)])
      if len(batch_x) >= batch_size:
        yield torch.Tensor(batch_x).to(device), torch.Tensor(batch_y).to(device)
        batch_x = []
        batch_y = []

def get_vocoder():
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    return vocoder



def main(device='cpu', batch_size=32):
    #things = torch.load('./models/mel-generator-model-500it.pth')
    wandb.init(project='MelGenerator')
    phoneme_classifier = PhonemeClassifier().to(device)
    phoneme_classifier.load_state_dict(torch.load('./phoneme_classifier/models/phoneme-classifier-final64-state_dict-30ep.pth'))

    #phoneme_classifier = torch.load('./phoneme_classifier/models/phoneme-classifier--final2-model-50ep.pth')['model']
    phoneme_classifier.eval()

    vocoder = get_vocoder()#torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

    model = MelGenerator().to(device)
    wandb.watch(model)
    #model.load_state_dict(torch.load('./models/mel-generator-state_dict-500it.pth'))
    lr = wandb.config.lr = 4e-4

    optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = things['optimizer']
    #sched = things['scheduler']

    sched = ReduceLROnPlateau(optimizer, factor=0.8, patience=300, threshold=0.000000001, verbose=True)
    criterion_post = nn.L1Loss()
    criterion = nn.L1Loss()
    torch.autograd.set_detect_anomaly(True)

    smooth_loss = 0.15
    batch_size = wandb.config.batch_size = batch_size
    seq_len = wandb.config.seq_len = 256
    fmax = wandb.fmax = 500
    k = wandb.k = 1
    counter = 0
    for ep in range(10000):
      for batch in get_batches(phoneme_classifier, vocoder, batch_size=batch_size, seq_len=seq_len, fmax=fmax, k=k, device=device):
        x, y = batch
        #x = torch.Tensor(x).to(device)
        #y = torch.Tensor(y).to(device)
        counter += 1
        hx = None
        retain_graph = True

        optimizer.zero_grad()
        logits, post_out = model.forward(x, hidden=hx)
        loss = criterion(logits, y)
        post_loss = criterion_post(post_out, y)
        full_loss = loss + post_loss

        #loss /= batch_size
        smooth_loss = smooth_loss*0.95 + loss.item()*0.05
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

        wandb.log({'loss': loss.item()**2, 'total_loss': full_loss.item()**2, 'post_loss': post_loss.item()**2})
        
        sched.step(loss.item())

        if counter % 25 == 0 or counter == 1:
          print('iter: {} loss: {}, smooth_loss: {}'.format(counter, loss.item(), smooth_loss))

          if counter % 3000 == 0 or counter == 1:
            saved_thing = {'model': model, 'optimizer': optimizer, 'scheduler': sched}
            torch.save(model.state_dict(), './models/mel-generator-state_dict-{}it.pth'.format(counter))
            torch.save(saved_thing, './models/mel-generator-model-{}it.pth'.format(counter))
            generate(model, phoneme_classifier, vocoder, name='{}it'.format(counter), device=device, k=k)
      print('batch {} done'.format(str(ep)))
  
if __name__ == '__main__':
    main(device='cuda', batch_size=32)
    
