import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchensemble.fusion import FusionClassifier
from torchensemble.voting import VotingClassifier
from torchensemble.bagging import BaggingClassifier
from torchensemble.gradient_boosting import GradientBoostingClassifier
from torchvision import models

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
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 20)

    def forward(self, input):
        return self.model(input)

# class Wide(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.resnext101_32x8d(pretrained=True)
#         self.model.fc = torch.nn.Linear(self.model.fc.in_features, 20)
#
#     def forward(self, input):
#         return self.model(input)


class Resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 20)

    def forward(self, input):
        return self.model(input)

class State:
    def __init__(self, model, optim):
        self.model = model
        self.optimizer = optim
        self.epoch = 0

model = models.wide_resnet101_2(pretrained=True)
state = State(model,None)
torch.save(state,"wideresnet101.pth")
