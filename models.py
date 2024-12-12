from torch import nn
from torchvision import models


class FineTunedResNet18(nn.Module):
    """
    Fine tuning a ResNet-18.
    """
    def __init__(self):
        super().__init__()

        # Load the pre-trained ResNet-34 model
        self.resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')

        self.resnet.conv1 = nn.Conv2d(
                    in_channels=1,  # grayscale
                    out_channels=self.resnet.conv1.out_channels,
                    kernel_size=self.resnet.conv1.kernel_size,
                    stride=self.resnet.conv1.stride,
                    padding=self.resnet.conv1.padding
                )

        # Freeze all parameters by default
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the fully connected layer with a new one
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 6)

        # Unfreeze the last 'fine_tune_layers' layers
        for param in self.resnet.layer4[-2:].parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        '''forward path'''
        return self.resnet(x)
