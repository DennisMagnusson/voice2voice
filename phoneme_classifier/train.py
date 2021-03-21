import torch
from torch import nn, optim
from read_data import get_data, get_dict

from tqdm import tqdm
from model import PhonemeClassifier

from random import shuffle
"""
class PhonemeClassifier(nn.Module):
  def __init__(self):
    super(PhonemeClassifier, self).__init__()
    self.linear0 = nn.Linear(80, 256)
    self.lstm1 = nn.GRU(256, 256, 3, dropout=0.15, batch_first=True, bidirectional=True)
    self.dropout = nn.Dropout(0.1)
    self.linear1 = nn.Linear(512, 256)
    self.linear2 = nn.Linear(256, 61)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x, hidden=None):
    out = self.linear0(x)
    out, hx = self.lstm1(out, hidden)
    #print(out.shape)
    out = self.dropout(out)
    out = self.linear1(out)
    out = self.linear2(out)
    #out = self.softmax(self.linear1(out))
    out = torch.transpose(out, 1, 2)
    return out, hx 
"""

def get_batches(mels, phones, batch_size=64, seq_len=64):
  batch_x = []
  batch_y = []
  # Shuffle data
  c = list(zip(mels, phones))
  shuffle(c)
  mels, phones = zip(*c)
  for mel, phone in tqdm(zip(mels, phones), total=len(mels)):
    for i in range(0, len(mel)-seq_len, seq_len):
      batch_x.append(mel[i:i+seq_len])
      batch_y.append(phone[i:i+seq_len])

      if len(batch_x) == seq_len:
        torch.Tensor(batch_x)
        yield torch.Tensor(batch_x), torch.LongTensor(batch_y)
        batch_x = []
        batch_y = []
  
  #yield torch.Tensor(batch_x), torch.LongTensor(batch_y)


if __name__ == '__main__':
    mels, phones = get_data('TIMIT-data/data/TRAIN/')
    #torch.set_num_threads(4)

    #things = torch.load('./models/phoneme-classifier-model.pth')
    things = torch.load('./models/phoneme-classifier--final64-model-20ep.pth')
    model = things['model']
    optimizer = things['optimizer']

    #model = PhonemeClassifier()
    #model.load_state_dict(torch.load('./models/phoneme-classifier-final64-state_dict-20ep.pth'))
    #optimizer = optim.Adam(model.parameters(), lr=0.0003)

    criterion = nn.CrossEntropyLoss()
    torch.autograd.set_detect_anomaly(True)
    seq_len = 64

    #smooth_acc = 1.0/61
    #smooth_loss = 0.0177
    smooth_acc = 0.770
    smooth_loss = 0.018
    batch_size = 32
    seq_len = 64
    for ep in range(21, 100):
      counter = 0
      #for mel, phone in tqdm(zip(mels, phones), total=len(mels)):
      for x, y in get_batches(mels, phones, batch_size=batch_size, seq_len=seq_len):
        counter += 1
        hx = None
        retain_graph = True

        optimizer.zero_grad()
        #logits, hx = model.forward(x, hidden=hx)
        logits, _ = model.forward(x, hidden=hx)
        #print(logits.shape)
        # TODO Might need a loop here
        loss = criterion(logits, y)
        
        # Get accuracy
        correct = 0
        for i in range(batch_size):
          correct += (torch.argmax(logits[i], dim=0) == y[i]).sum().item()

        #accuracy = float(correct) / len(mel)
        accuracy = float(correct) / (batch_size*seq_len)
        smooth_acc = smooth_acc*0.95 + accuracy*0.05

        # TODO Might be wrong if last seq is shorter
        loss /= batch_size
        smooth_loss = smooth_loss*0.95 + loss.item()*0.05
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

        #if counter % 500 == 0 or counter == 1:
          #print('smooth_accuracy: {}\tsmooth_loss: {}'.format(smooth_acc, smooth_loss))
          #print('accuracy: {}\tloss: {}'.format(accuracy, loss.item()))
      print('ep {} done'.format(ep))
      print('smooth_accuracy: {}\tsmooth_loss: {}'.format(smooth_acc, smooth_loss))
      print('accuracy: {}\tloss: {}'.format(accuracy, loss.item()))
      
      if ep % 10 == 0:
        saved_thing = {'model': model, 'optimizer': optimizer}
        torch.save(model.state_dict(), './models/phoneme-classifier-final64-state_dict-{}ep.pth'.format(ep))
        torch.save(saved_thing, './models/phoneme-classifier--final64-model-{}ep.pth'.format(ep))
        print('checkpoint saved')
