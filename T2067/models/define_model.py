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
from torchvision import datasets, transforms
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


class CreateModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes = num_classes)

    def forward(self,x):
        x = self.model(x)
        return x

