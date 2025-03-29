import torch
import torch.nn as nn
import torchvision.models as models

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=1000, hidden_dim=512, num_layers=2):
        super(CNN_LSTM, self).__init__()

        # Load ResNet18 without the final classification layer
        resnet = models.resnet18()
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
        self.feature_dim = resnet.fc.in_features  # Feature size from ResNet18

        # Define LSTM
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.3)

        # Final classification layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape  # (Batch, Time, Channels, Height, Width)

        # Reshape for CNN: Process (B*T, C, H, W) through ResNet18
        x = x.view(B * T, C, H, W)  # Merge batch & time
        x = self.feature_extractor(x)  # Pass through ResNet18 (CNN)
        x = x.view(B, T, -1)  # Reshape to (B, T, Feature_dim)

        # Pass through LSTM
        x, _ = self.lstm(x)  # (B, T, Hidden_dim)

        # Classification (take last time step output)
        x = self.fc(x[:, -1, :])  # Use last LSTM output for prediction

        return x