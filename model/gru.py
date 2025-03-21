import torch
from torch import nn


class GRU(nn.Module):
    def __init__(self, vocab_size=1000, frame_count=64, input_channels=3, image_size=224, hidden_dim=128):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2))
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))

        self.gru = nn.GRU(input_size=128 * (image_size // 8) * (image_size // 8), hidden_size=hidden_dim, batch_first=True)
    
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(256, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Reshape to match GRU
        batch_size, channels, frames, height, width = x.size()
        # x = x.view(batch_size, frames, (channels * height * width))
        x = x.view(batch_size, frames, -1)      # Shape: (batch_size, frames, channels * height * width)

        _, h = self.gru(x)
        x = h.squeeze(0)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.softmax(x)

        return x
