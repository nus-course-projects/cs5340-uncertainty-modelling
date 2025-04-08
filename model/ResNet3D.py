import torch.nn as nn
from torchvision.models.video import r3d_18, r2plus1d_18, mc3_18
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

class ResNet3D(nn.Module):
    def __init__(self, num_classes, frozen_layers=None, bayesian_layers=None, bayesian_options=None):
        """
        Args:
            num_classes (int): Number of output classes.
            frozen_layers (int or None): 
                - If an integer between 0 and 4, the model uses pretrained weights and that many layers will be frozen.
                - If None, the model is initialized without pretrained weights.
        """
        super(ResNet3D, self).__init__()
        if frozen_layers is not None:
            # Use pretrained weights, which expect 400 classes
            self.model = r2plus1d_18(pretrained=True, progress=True)
        else:
            # Initialize model without pretrained weights and with desired num_classes directly.
            self.model = r2plus1d_18(pretrained=False, progress=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(            
            nn.Linear(in_features, num_classes)
        )

        # Freeze layers if frozen_layers is specified
        if frozen_layers is not None:
            if not (0 <= frozen_layers <= 5):
                raise ValueError("frozen_layers must be between 0 and 5.")
            # Freeze the stem (initial convolutional layers) if at least 1 layer is to be frozen.
            if frozen_layers >= 1:
                for param in self.model.stem.parameters():
                    param.requires_grad = False
            # Freeze layer1 if at least 2 layers are to be frozen.
            if frozen_layers >= 2:
                for param in self.model.layer1.parameters():
                    param.requires_grad = False
            # Freeze layer2 if at least 3 layers are to be frozen.
            if frozen_layers >= 3:
                for param in self.model.layer2.parameters():
                    param.requires_grad = False
            # Freeze layer3 if frozen_layers is 4.
            if frozen_layers >= 4:
                for param in self.model.layer3.parameters():
                    param.requires_grad = False
            # Freeze layer4 if frozen_layers is 5.
            if frozen_layers >= 5:
                for param in self.model.layer4.parameters():
                    param.requires_grad = False

        if bayesian_layers is not None:
            if bayesian_options is None:
                raise ValueError("bayesian_options must be provided if bayesian_layers is not None.")
            if not (0 <= bayesian_layers <= 6):
                raise ValueError("bayesian_layers must be between 0 and 6.")
            if bayesian_layers <= 1:
                dnn_to_bnn(self.model.stem, bayesian_options)
            if bayesian_layers <= 2:
                dnn_to_bnn(self.model.layer1, bayesian_options)
            if bayesian_layers <= 3:
                dnn_to_bnn(self.model.layer2, bayesian_options)
            if bayesian_layers <= 4:
                dnn_to_bnn(self.model.layer3, bayesian_options)
            if bayesian_layers <= 5:
                dnn_to_bnn(self.model.layer4, bayesian_options)
            if bayesian_layers <= 6:
                dnn_to_bnn(self.model.fc, bayesian_options)
                
    def forward(self, x):
        return self.model(x)