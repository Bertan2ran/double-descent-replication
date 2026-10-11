import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=1):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool_size > 1:
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class CNN5(nn.Module):
    def __init__(self, k: int, num_classes: int = 10):
        super().__init__()
        self.k = k
        # MaxPool strides from paper [1, 2, 2, 8]
        self.layer1 = ConvBNReLU(3, k, pool_size=1)       # 32x32 t0 32x32
        self.layer2 = ConvBNReLU(k, 2 * k, pool_size=2)   # 32x32 to 16x16
        self.layer3 = ConvBNReLU(2 * k, 4 * k, pool_size=2) # 16x16 to 8x8
        self.layer4 = ConvBNReLU(4 * k, 8 * k, pool_size=8) # 8x8 to 1x1
        self.fc = nn.Linear(8 * k, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out