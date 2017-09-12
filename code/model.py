import torch
from torch import nn

class CompoundLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CompoundLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        #in_channels: 3 for colored images(RGB), 1 for grayscale
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        #Batch normalization is a technique that can improve the learning rate of a neural network.
        #It does so by minimizing internal covariate shift which is essentially the phenomenon of
        #each layer's input distribution changing as the parameters of the layer above it change during training
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output


class Model(nn.Module):
    def __init__(self, num_class):
        super(Model, self).__init__()

        self.CompoundLayer1 = CompoundLayer(in_channels=3, out_channels=32)
        self.CompoundLayer2 = CompoundLayer(in_channels=32, out_channels=32)
        self.CompoundLayer3 = CompoundLayer(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.CompoundLayer4 = CompoundLayer(in_channels=32, out_channels=64)
        self.CompoundLayer5 = CompoundLayer(in_channels=64, out_channels=64)
        self.CompoundLayer6 = CompoundLayer(in_channels=64, out_channels=64)
        self.CompoundLayer7 = CompoundLayer(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.CompoundLayer8 = CompoundLayer(in_channels=64, out_channels=128)
        self.CompoundLayer9 = CompoundLayer(in_channels=128, out_channels=128)
        self.CompoundLayer10 = CompoundLayer(in_channels=128, out_channels=128)
        self.CompoundLayer11 = CompoundLayer(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.CompoundLayer12 = CompoundLayer(in_channels=128, out_channels=128)
        self.CompoundLayer13 = CompoundLayer(in_channels=128, out_channels=128)
        self.CompoundLayer14 = CompoundLayer(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        self.net = nn.Sequential(
            self.CompoundLayer1, 
            self.CompoundLayer2, 
            self.CompoundLayer3, 
            self.pool1, 
            self.CompoundLayer4, 
            self.CompoundLayer5, 
            self.CompoundLayer6, 
            self.CompoundLayer7, 
            self.pool2, 
            self.CompoundLayer8, 
            self.CompoundLayer9, 
            self.CompoundLayer10, 
            self.CompoundLayer11, 
            self.pool3,
            self.CompoundLayer12, 
            self.CompoundLayer13, 
            self.CompoundLayer14, 
            self.avgpool
        )

        self.fc = nn.Linear(in_features=128, out_features=num_class)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 128)
        output = self.fc(output)
        return output
