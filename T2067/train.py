import sys
import os
current_path = os.path.join(os.getcwd(),'T2067')
sys.path.append(current_path)

from models.define_model import CreateModel
from models.training import *
from utils.crop import *
from utils.dataloader import *
from utils.process_data import *


dataframe = create_train_dataframe('data/train','cropped')
model = CreateModel("efficientnet_b3",num_classes=18)

folded_data = split_df(dataframe, kfold_n = 5)

for fold in range(5):
    train_data = folded_data[fold][0]
    valid_data = folded_data[fold][1]