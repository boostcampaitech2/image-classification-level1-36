import os
import pandas as pd
import re
import cv2
import numpy as np
### Data Loader ###

import os
import pandas as pd
import re
import cv2
import numpy as np
import timm
import os
import time
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam, AdamW
import gc
from tqdm import notebook
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import albumentations
import albumentations.pytorch
from sklearn.metrics import f1_score

log_steps = 20


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_len = len(train_loader)
    corrects = 0
    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")
        _, preds = torch.max(outputs,1)
        corrects += torch.sum(preds == labels.data)
    acc = corrects / train_len

    return acc