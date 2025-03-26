import torch
import torch.nn as nn
import torch.nn.functional as F

class KeyPointsTransformer(nn.Module):
    def __init__(self, num_keypoints=133, feature_dim=3, hidden_size=256, 
                 num_heads=8, num_layers=4, dropout=0.1, num_classes=1000):
        """
        KeyPoints Transformer network for processing human pose keypoints data.
        
        Args:
            num_keypoints (int): Number of keypoints in each frame (default: 133)
            feature_dim (int): Dimension of each keypoint (default: 3 for x, y, score)
            hidden_size (int): Size of transformer hidden state
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout (float): Dropout probability
            num_classes (int): Number of output classes
        """
        super(KeyPointsTransformer, self).__init__()
        
        self.num_keypoints = num_keypoints
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        
        # Keypoint embedding - transforms each keypoint's features
        self.keypoint_embedding = nn.Linear(feature_dim, hidden_size)
        
        # Keypoint position embedding - using Embedding layer instead of Parameter
        self.keypoint_pos_embedding = nn.Embedding(num_keypoints, hidden_size)
        
        # Frame position embedding - learns a representation for each frame position
        self.frame_pos_embedding = nn.Embedding(1000, hidden_size)  # Max 1000 frames
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_keypoints, 3)
                              where 3 represents (x, y, score) for each keypoint
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, num_keypoints, feature_dim = x.size()
        
        # Process each frame's keypoints
        # Reshape to (batch_size * seq_len, num_keypoints, feature_dim)
        x_reshaped = x.view(batch_size * seq_len, num_keypoints, feature_dim)
        
        # Apply keypoint embedding to each keypoint
        x_embedded = self.keypoint_embedding(x_reshaped)  # (batch_size * seq_len, num_keypoints, hidden_size)
        
        # Add keypoint position embeddings using indices
        keypoint_indices = torch.arange(num_keypoints, device=x.device)
        keypoint_pos_embed = self.keypoint_pos_embedding(keypoint_indices)  # (num_keypoints, hidden_size)
        
        # Add keypoint position embeddings to each frame
        x_embedded = x_embedded + keypoint_pos_embed.unsqueeze(0)
        
        # Reshape back to include sequence dimension
        x_embedded = x_embedded.view(batch_size, seq_len, num_keypoints, self.hidden_size)
        
        # Add frame position embeddings
        frame_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        frame_embeddings = self.frame_pos_embedding(frame_positions)  # (batch_size, seq_len, hidden_size)
        
        # Add frame embeddings to each keypoint in the corresponding frame
        frame_embeddings = frame_embeddings.unsqueeze(2).expand(-1, -1, num_keypoints, -1)
        x_embedded = x_embedded + frame_embeddings
        
        # Reshape to (batch_size, seq_len * num_keypoints, hidden_size) for transformer
        x_embedded = x_embedded.view(batch_size, seq_len * num_keypoints, self.hidden_size)
        
        # Apply layer normalization
        x_normalized = self.layer_norm(x_embedded)
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(x_normalized)
        
        # Global average pooling over all keypoints and frames
        pooled_output = transformer_output.mean(dim=1)  # (batch_size, hidden_size)
        
        # Classification
        output = self.classifier(pooled_output)
        
        return output

class KeyPointsLSTM(nn.Module):
    def __init__(self, input_size=133*3, hidden_size=256, mlp_hidden_size=512, num_layers=2, dropout=0.5, num_classes=1000):
        """
        KeyPoints LSTM network for processing human pose keypoints data.
        
        Args:
            input_size (int): Size of input features (default: 133*3 for 133 keypoints with x, y, score)
            hidden_size (int): Size of LSTM hidden state
            mlp_hidden_size (int): Size of MLP hidden layers
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
            num_classes (int): Number of output classes
        """
        super(KeyPointsLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # MLP layers before LSTM to process each frame's keypoints
        self.pre_mlp = nn.Sequential(
            nn.Linear(input_size, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, mlp_hidden_size // 2),
            nn.BatchNorm1d(mlp_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=mlp_hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_keypoints, 3)
                              where 3 represents (x, y, score) for each keypoint
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # print(x.shape)
        batch_size, seq_len, num_keypoints, features = x.size()
        
        # Reshape to (batch_size, seq_len, num_keypoints * features)
        x = x.view(batch_size, seq_len, -1)
        
        # Process each frame with MLP
        # Reshape to process all frames at once
        x_reshaped = x.view(batch_size * seq_len, -1)
        x_processed = self.pre_mlp(x_reshaped)
        x = x_processed.view(batch_size, seq_len, -1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
