"""
VGG16 Implementation (PyTorch)

- Feature extractor: ConvBlock modules
- Classifier: FCOnlyLayer modules
- Extra modules included for modularity: FlattenLayer, MaxPoolLayer
- Model architecture corresponds to VGG16:
    ConvBlock(3, 64, 2) -> MaxPool
    ConvBlock(64, 128, 2) -> MaxPool
    ConvBlock(128, 256, 3) -> MaxPool
    ConvBlock(256, 512, 3) -> MaxPool
    ConvBlock(512, 512, 3) -> MaxPool
    FC layers: 4096 -> 4096 -> num_classes
"""

import torch
import torch.nn as nn
from conv_block import ConvBlock
from fc_only_layer import FCOnlyLayer

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            ConvBlock(3, 64, 2),     
            ConvBlock(64, 128, 2),
            ConvBlock(128, 256, 3),
            ConvBlock(256, 512, 3),
            ConvBlock(512, 512, 3)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),            
            FCOnlyLayer(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            FCOnlyLayer(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            FCOnlyLayer(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
