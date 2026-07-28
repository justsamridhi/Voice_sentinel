import torch
import torch.nn as nn
import torchvision.models as models


class CNNBaseline(nn.Module):
    """A lightweight 2D Convolutional Neural Network baseline for anti-spoofing."""

    def __init__(self, num_classes: int = 2):
        """Initializes baseline CNN layers."""
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of CNNBaseline.

        Args:
            x: Input feature maps of shape (batch, 1, freq_bins, time_steps).

        Returns:
            torch.Tensor: Logits tensor of shape (batch, num_classes).
        """
        feats = self.features(x)
        return self.classifier(feats)


class ResNet34Baseline(nn.Module):
    """ResNet34 adapted for single-channel audio feature inputs."""

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        """Initializes ResNet34 and modifies input convolutional layer.

        Args:
            num_classes: Number of classification targets.
            pretrained: Whether to load ImageNet pre-trained weights.
        """
        super().__init__()
        # Load torchvision resnet34 model
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet34(weights=weights)

        # Replace first conv layer: change input channels from 3 (RGB) to 1 (Spectrogram/LFCC)
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        # Replace final fully connected layer
        self.resnet.fc = nn.Linear(
            in_features=self.resnet.fc.in_features,
            out_features=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ResNet34Baseline.

        Args:
            x: Input feature maps of shape (batch, 1, freq_bins, time_steps).

        Returns:
            torch.Tensor: Logits tensor of shape (batch, num_classes).
        """
        return self.resnet(x)
