import torch
from torch import nn, optim
import torch.nn.functional as F

class PhonemeClassifier(nn.Module):
  def __init__(self):
    super(PhonemeClassifier, self).__init__()
    self.linear0 = nn.Linear(80, 256)
    self.bn0 = nn.BatchNorm1d(256)
    self.lstm1 = nn.GRU(256, 256, 2, dropout=0.15, batch_first=True, bidirectional=True)
    self.bn1 = nn.BatchNorm1d(512)
    self.dropout = nn.Dropout(0.1)
    self.linear1 = nn.Linear(512, 256)
    self.bn2 = nn.BatchNorm1d(256)
    self.linear2 = nn.Linear(256, 61)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x, hidden=None):
    # l1
    out = self.linear0(x)
    out = self.bn0(out.transpose(1, 2)).transpose(1, 2)

    # rnn
    out, hx = self.lstm1(out, hidden)
    out = self.bn1(out.transpose(1, 2)).transpose(1, 2)

    # out0
    out = self.dropout(out)
    out = self.linear1(out)
    out = torch.sigmoid(out)
    out = self.bn2(out.transpose(1, 2)).transpose(1, 2)

    # out1
    out = self.linear2(out)
    #out = self.softmax(self.linear1(out))
    out = torch.transpose(out, 1, 2)
    return out, hx 

