import torch
import torch.nn as nn
import torchvision.models as mod

class MyModel(nn.Module): # resnet 18
    def __init__(self, num_classes):
        super().__init__()
        self.mod1 = mod.resnet18(pretrained=True)
        self.mod1.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    def forward(self, x):
        x=self.mod1(x)
        return x
