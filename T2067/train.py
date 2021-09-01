#from utils import crop, dataloader, process_data
import sys
import os
current_path = os.path.join(os.getcwd(),'T2067')
sys.path.append(current_path)

from models.define_model import CreateModel
from utils.crop import *
from utils.dataloader import *
from utils.process_data import *

dataframe = create_train_dataframe('data/train','cropped')
model = CreateModel("efficientnet_b3",num_classes=18)
