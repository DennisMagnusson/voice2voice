import torch
from torch import nn, optim
from read_data import get_data, get_dict

from tqdm import tqdm

class PhonemeClassifier(nn.Module):
  def __init__(self):
    super(PhonemeClassifier, self).__init__()
    self.lstm1 = nn.GRU(80, 128, 2, dropout=0.2, batch_first=True, bidirectional=True)
    self.dropout = nn.Dropout(0.1)
    self.linear1 = nn.Linear(256, 128)
    self.linear2 = nn.Linear(128, 61)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x, hidden=None):
    out, hx = self.lstm1(x, hidden)
    #print(out.shape)
    out = self.dropout(out)
    out = self.linear1(out)
    out = self.linear2(out)
    #out = self.softmax(self.linear1(out))
    out = torch.transpose(out, 1, 2)
    return out, hx 

if __name__ == '__main__':
    mels, phones = get_data('TIMIT-data/data/TRAIN/')

    model = PhonemeClassifier()
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    torch.autograd.set_detect_anomaly(True)
    seq_len = 32

    smooth_acc = 1.0/61
    smooth_loss = 0.15
    for ep in range(100):
      for mel, phone in tqdm(zip(mels, phones), total=len(mels)):
        loss = 0
        batch_size = 0
        hx = None
        retain_graph = True

        correct = 0

        for i in range(0, len(mel), seq_len):
          x = torch.Tensor([mel[i:i+seq_len]])
          y = torch.LongTensor([phone[i:i+seq_len]])

          optimizer.zero_grad()
          logits, hx = model.forward(x, hidden=hx)
          #print(logits.shape)
          # TODO Might need a loop here
          loss += criterion(logits, y)
          
          #print(torch.argmax(logits[0], dim=0))
          #print(y[0])
          correct += (torch.argmax(logits[0], dim=0) == y[0]).sum().item()
          #retain_graph = False
          #optimizer.step()
          batch_size += 1
       
        accuracy = float(correct) / len(mel)
        smooth_acc = smooth_acc*0.9 + accuracy*0.1

        # TODO Might be wrong if last seq is shorter
        loss /= len(mel)
        smooth_loss = smooth_loss*0.9 + loss.item()*0.1
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

      print('smooth_accuracy: {}\tsmooth_loss: {}'.format(smooth_acc, smooth_loss))
      print('accuracy: {}\tloss: {}'.format(accuracy, loss.item()))
