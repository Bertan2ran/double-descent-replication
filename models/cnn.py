import torch
import torch.nn as nn

class FiveLayerCNN(nn.Module):
    def __init__(self, k=128, num_classes=10):
        super(FiveLayerCNN, self).__init__()
        self.features = nn.Sequential(
            self._make_layer(3, k, pool_stride=1),
            self._make_layer(k, 2*k, pool_stride=2),
            self._make_layer(2*k, 4*k, pool_stride=2),
            self._make_layer(4*k, 8*k, pool_stride=8)
        )
        self.classifier = nn.Linear(8*k, num_classes)

    def _make_layer(self, in_channels, out_channels, pool_stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_stride, stride=pool_stride)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)