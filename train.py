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

from sys import argv


class MelGenerator(nn.Module):
  def __init__(self):
    super(MelGenerator, self).__init__()
    self.enclinear0 = nn.Linear(62, 256)
    self.encdropout0 = nn.Dropout(0.2)
    self.enclinear1 = nn.Linear(256, 128)
    self.encdropout1 = nn.Dropout(0.2)

    #self.linear0 = nn.Linear(256, 128)
    #self.cbhg0 = CBHG(128, 8)

    conv_size = 256   

    self.conv0 = ConvThingy(128, 128, 7, nn.ReLU())
    self.conv1 = ConvThingy(128, conv_size, 7, nn.ReLU())
    self.linear1 = nn.Linear(conv_size, conv_size)
    self.pool = nn.AvgPool1d(4, stride=4)

    self.convies = []
    for i in range(7):
      self.convies.append(ConvThingy(conv_size, conv_size, 5, nn.ReLU()))
    
    self.convies = nn.ModuleList(self.convies)

    self.declinear0 = nn.Linear(256, 256)
    self.declinear1 = nn.Linear(256, 80)


    conv2d_size = 32
    self.convdd0 = Conv2DThingy(1, conv2d_size, 5, nn.ReLU())
    self.convies2d = []
    for i in range(4):
      self.convies2d.append(Conv2DThingy(conv2d_size, conv2d_size, 5, nn.ReLU()))
    self.convies2d = nn.ModuleList(self.convies2d)
    self.convdd10 = Conv2DThingy(conv2d_size, 1, 5, None)

    #self.convdd10 = Conv2DThingy(256, 1, 5, nn.ReLU())

    

  def forward(self, x):
    out = F.relu(self.encdropout0(self.enclinear0(x)))
    out = F.relu(self.encdropout1(self.enclinear1(out)))

    #out = self.cbhg0(out)
    #out = F.relu(self.linear0(out))

    out = out.transpose(1, 2)
    out = self.conv0(out)
    out = self.conv1(out)
    out = self.pool(out)
    for c in self.convies:
      out = c(out) + out

    out = out.transpose(1, 2)
    out = F.relu(self.declinear0(out))
    out = self.declinear1(out)

    out = out.unsqueeze(1)
    out = self.convdd0(out)
    for c in self.convies2d:
      out = c(out) + out

    out = self.convdd10(out)
    out = out.squeeze(1)

    out = 3*F.hardsigmoid(out)

    return out, 0

class Discriminator(nn.Module):
  def __init__(self, seq_len, n_mels=80):
    super(Discriminator, self).__init__()
    # 80 x 256 x 1
    self.conv0 = Conv2DThingy(1, 64, 5, nn.LeakyReLU(0.2), stride=2)
    # 40 x 128 x 64
    self.conv1 = Conv2DThingy(64, 64, 5, nn.LeakyReLU(0.2), stride=2)
    # 20 x 64 x 128
    self.conv2 = Conv2DThingy(64, 128, 5, nn.LeakyReLU(0.2), stride=2)
    # 10 x 32 x 256
    self.conv3 = Conv2DThingy(128, 256, 5, nn.LeakyReLU(0.2), stride=2)
    # 5 x 16 x 256
    self.conv4 = Conv2DThingy(256, 1, (seq_len//16, 5), None, stride=2)

    self.linear = nn.Linear(5*16*256, 1, bias=False)

  def forward(self, x):
    out = x.unsqueeze(1)
    out = self.conv0(out)
    out = self.conv1(out)
    out = self.conv2(out)
    out = self.conv3(out)
    #out = self.conv4(out)
    out = out.flatten(start_dim=1)
    out = self.linear(out)
    return torch.sigmoid(out)

class Conv2DThingy(nn.Module):
  def __init__(self, in_size, out_size, kernel_size, activation, stride=1):
    super(Conv2DThingy, self).__init__()
    self.bn = nn.BatchNorm2d(out_size)
    self.activation = activation
    if type(kernel_size) == tuple:
      padding = 0
    else:
      padding = kernel_size//2

    self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=padding, stride=stride)

  def forward(self, x):
    #out = self.pad(x)
    out = self.conv(x)
    out = self.bn(out)
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

def generate(model, phoneme_classifier, vocoder, name='1it', device='cpu', k=1):
  files = [a for a in ['LJ042-0033.wav', 'LJ008-0195.wav', 'LJ050-0174.wav']]
  model.eval()
  phoneme_classifier.eval()
  try:
    mkdir('./generated/{}'.format(name))
  except FileExistsError:
    pass

  for filename in files:
    wav, sr = get_wav('./speech_data/LJSpeech-1.1/wavs/'+filename)
    mel = wav_to_mel(wav, sr).T
    mel_vocoder = vocoder(torch.Tensor([wav]))[0].T
    #f0, prob = get_pyin(wav, sr)
    f0 = torch.Tensor(get_pyin(wav, sr, k=k)).to(device)
    with torch.no_grad():
      phones, _ = phoneme_classifier(torch.Tensor([mel]).to(device))
      phones = phones.to(device)
      #inpu = torch.cat((phones.transpose(1, 2), torch.Tensor(f0), torch.Tensor(prob)), 1)
      inpu = torch.cat((phones.transpose(1, 2), f0.unsqueeze(0).unsqueeze(2)), 2)
      mel2, _ = model(inpu)
      
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

    smooth_loss = 0.15
    batch_size = wandb.config.batch_size = batch_size
    seq_len = wandb.config.seq_len = 256
    fmax = wandb.fmax = 500
    k = wandb.k = 1
    counter = 0


    #phoneme_classifier = torch.load('./phoneme_classifier/models/phoneme-classifier--final2-model-50ep.pth')['model']
    phoneme_classifier.eval()

    vocoder = get_vocoder()#torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

    model = MelGenerator().to(device)
    disc = Discriminator(seq_len)
    wandb.watch(model)
    #model.load_state_dict(torch.load('./models/mel-generator-state_dict-500it.pth'))
    #lr = wandb.config.lr = 4e-4
    lr = wandb.config.lr = 0.001


    optimizer = optim.Adam(model.parameters(), lr=lr)
    d_optimizer = optim.Adam(disc.parameters(), lr=lr)
    g_optimizer = optim.Adam(model.parameters(), lr=lr)#XXX Dont know if this can be same as first optimizer?

    #optimizer = things['optimizer']
    #sched = things['scheduler']

    sched = ReduceLROnPlateau(optimizer, factor=0.8, patience=300, threshold=0.000000001, verbose=True)
    #criterion = nn.MSELoss()
    gen_criterion = nn.L1Loss()
    criterion = nn.BCELoss()
    #criterion = nn.MSELoss()
    #torch.autograd.set_detect_anomaly(True)

    for ep in range(10000):
      for batch in get_batches(phoneme_classifier, vocoder, batch_size=batch_size, seq_len=seq_len, fmax=fmax, k=k, device=device):
        x, y = batch
        counter += 1
        retain_graph = True

        if counter < 50:# Pretrain generator
          model.zero_grad()
          fake_mel, _ = model.forward(x)
          loss = gen_criterion(fake_mel, y)

          #loss /= batch_size
          smooth_loss = smooth_loss*0.95 + loss.item()*0.05
          loss.backward()
          optimizer.step()
          
          if counter > 10:
            disc.zero_grad()
            #d_optimizer.zero_grad()
            outputs = disc.forward(y)
            d_loss_real = criterion(outputs, torch.ones_like(outputs))
            d_loss_real.backward()
            # Train on fakes
            fake_mel, _ = model.forward(x)
            outputs = disc.forward(fake_mel.detach()).view(-1)
            d_loss_fake = criterion(outputs, torch.zeros_like(outputs))
            d_loss_fake.backward()
            d_optimizer.step()

          wandb.log({'loss': loss.item()**2})
        else:# Train everything
          # Train discriminator
          # Train on reals
          disc.zero_grad()
          outputs = disc.forward(y)
          d_loss_real = criterion(outputs, torch.ones_like(batch_size))
          d_loss_real.backward()
          # Train on fakes
          fake_mels, _ = model.forward(x)
          outputs = disc.forward(fake_mels.detach())
          d_loss_fake = criterion(outputs, torch.zeros_like(batch_size))
          d_loss_fake.backward()
          d_optimizer.step()
          # Train generator
          model.zero_grad()

          outputs = disc.forward(fake_mels)
          g_loss = criterion(outputs, torch.ones_like(outputs))
          gen_loss = gen_criterion(fake_mels, y)
          tot_loss = g_loss + gen_loss
          tot_loss.backward()
          g_optimizer.step()
          print(outputs)

          di = {'d_loss':(d_loss_real.mean().item() + d_loss_fake.mean().item())/2.0,
                     'g_loss': g_loss.mean().item(),
                     'loss': gen_loss.mean().item()**2}

          wandb.log(di)
          print(di)

          pass

        
        #sched.step(loss.item())

        if counter % 25 == 0 or counter == 1:
          print('iter: {} loss: {}, smooth_loss: {}'.format(counter, loss.item(), smooth_loss))

          if counter % 3000 == 0 or counter == 1:
            saved_thing = {'model': model, 'optimizer': optimizer, 'scheduler': sched}
            torch.save(model.state_dict(), './models/mel-generator-state_dict-{}it.pth'.format(counter))
            torch.save(saved_thing, './models/mel-generator-model-{}it.pth'.format(counter))
            #generate(model, phoneme_classifier, vocoder, name='{}it'.format(counter), device=device, k=k)
      print('batch {} done'.format(str(ep)))
  
if __name__ == '__main__':
    if len(argv) == 1:
      main(device='cpu', batch_size=4)
    else:
      main(device=argv[1], batch_size=int(argv[2]))
    
