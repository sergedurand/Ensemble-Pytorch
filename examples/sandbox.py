from torch import nn
from torchvision import models
import torch


class Mobilenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 20)

    def forward(self, input):
        return self.model(input)


class Resnext(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnext101_32x8d(pretrained=True)
        self.model.fc = torch.nn.Linear(2048, 20)

    def forward(self, input):
        return self.model(input)


class Resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.re(pretrained=True)
        self.model.fc = torch.nn.Linear(2048, 20)

    def forward(self, input):
        return self.model(input)

model = models.resnext101_32x8d(pretrained=True)
print(type(model))
