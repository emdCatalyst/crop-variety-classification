# MODEL

import torch
import torch.nn as nn
class CNNEncoder(nn.Module):
    def __init__(self, in_ch=13, out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128, out), nn.ReLU())

    def forward(self, x):
        return self.fc(self.net(x))


class CropCNNLSTM(nn.Module):
    def __init__(self, num_classes, in_ch=13, cnn_out=64, lstm_hidden=128, dropout=0.3):
        super().__init__()
        self.cnn  = CNNEncoder(in_ch, cnn_out)
        self.lstm = nn.LSTM(cnn_out, lstm_hidden, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.attn = nn.Linear(lstm_hidden, 1)   # learns date importance
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats   = self.cnn(x.view(B*T, C, H, W)).view(B, T, -1)
        out, _  = self.lstm(feats)                      # (B, T, hidden)
        weights = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        context = (weights * out).sum(dim=1)            # weighted avg over dates
        return self.head(context)
