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

    def __name__(self):
        return self.model.__class__.__name__


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
    def __name__(self):
        return self.model.__class__.__name__

class State:
    def __init__(self, model, optim):
        self.model = model
        self.optimizer = optim
        self.epoch = 0

est1 = Resnet50()
est2 = Mobilenet()
estimators = [est1, est2]

# Hyper-parameters
n_estimators = 4
output_dim = 20
lr = 1e-4
weight_decay = 5e-4
epochs = 10
resolution = 128

# Utils
batch_size = 8
data_dir = "bird_dataset/"  # MODIFY THIS IF YOU WANT
records = []
torch.manual_seed(0)

model = FusionClassifier(
    estimators=estimators,
    cuda=False,
    n_estimators=n_estimators,
    output_dim=output_dim,
    lr=lr,
    weight_decay=weight_decay,
    epochs=epochs,
)

print(model)
