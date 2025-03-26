import torch.nn as nn
from torchvision.models.video import r3d_18, r2plus1d_18

class ResNet3D(nn.Module):
    def __init__(self, num_classes, frozen_layers=None):
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
            base_model = r2plus1d_18(pretrained=True, progress=True)
            # Replace the final fully connected layer to match num_classes
            in_features = base_model.fc.in_features
            base_model.fc = nn.Linear(in_features, num_classes)
            self.model = base_model
        else:
            # Initialize model without pretrained weights and with desired num_classes directly.
            self.model = r2plus1d_18(pretrained=False, progress=True, num_classes=num_classes)

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

    def forward(self, x):
        return self.model(x)