#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils.dataset import load_msasl

label_threshold = 100
test_dataset, train_dataset, validation_dataset = load_msasl("bin", label_threshold)


# In[2]:


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


# In[3]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# In[4]:


cnn_lstm_model = CNN_LSTM().to(device)
print(cnn_lstm_model)
cnn_lstm_model.to(device)


# In[5]:


from torch.utils.data import DataLoader

data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
for videos, labels, metadata in data_loader:
        print(f"Batch of videos: {videos.shape}") # (batch_size, 64, C, H, W)
        print(f"Batch of labels: {labels.shape}") # (batch_size,)
        print(f"Metadata sample: {metadata}") # Dictionary of metadata
        print(labels)
        break # Checking the first batch

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)


# In[6]:


validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=True, num_workers=4)

for videos, labels, metadata in validation_loader:
        print(f"Batch of videos: {videos.shape}") # (B, T, C, H, W)
        print(f"Batch of labels: {labels.shape}") # (batch_size,)
        print(f"Metadata sample: {metadata}") # Dictionary of metadata
        print(labels)
        break # Checking the first batch


# In[7]:


import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm  # Import tqdm for progress bar

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer (Adam, no learning rate scheduler)
optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=0.001)

# Define scheduler (StepLR: reduces LR by gamma=0.1 every 5 epochs)
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 12  # Adjust based on performance

def check_nan(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"⚠️ NaN or Inf detected in {name}!")

for epoch in range(num_epochs):
    cnn_lstm_model.train()  # Set model to training mode
    total_loss = 0

    # Wrap data_loader with tqdm for training progress
    train_loader = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for videos, labels, metadata in train_loader:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()  # Reset gradients
        videos = videos.float() / 255.0
        outputs = cnn_lstm_model(videos)  # Forward pass


        for name, param in cnn_lstm_model.named_parameters():
            check_nan(param, f"Param {name}")

        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation

        for name, param in cnn_lstm_model.named_parameters():
            if param.grad is not None:
                check_nan(param.grad, f"Grad {name}")

        torch.nn.utils.clip_grad_norm_(cnn_lstm_model.parameters(), max_norm=5) # Gradient clipping

        optimizer.step()  # Update model weights

        total_loss += loss.item()
        train_loader.set_postfix(loss=loss.item())  # Update tqdm display

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Step the scheduler at the end of each epoch
    scheduler.step()
    print(f"New Learning Rate: {scheduler.get_last_lr()}")

    # Validation step
    cnn_lstm_model.eval()
    correct, total = 0, 0

    # Wrap validation loader with tqdm for validation progress
    val_loader = tqdm(validation_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for videos, labels, metadata in val_loader:
            videos = videos.to(device).float()  # Convert videos to float32
            labels = labels.to(device).long()   # Convert labels to long
            outputs = cnn_lstm_model(videos)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

