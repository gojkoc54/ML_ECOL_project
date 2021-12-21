import os
PROJECT_ROOT = '/workspace/ml_ecol_project'
os.chdir(PROJECT_ROOT)

DATASET_PATH = os.path.join(PROJECT_ROOT, '../labeled_data')
CHECKPOINTS_PATH = os.path.join(PROJECT_ROOT, 'checkpoints')
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

from utils import *
from models import *

import torch.nn as nn
import torchvision.models as models


if __name__ == '__main__':

    EPOCHS = 100
    LEARNING_RATE = 0.001
    MAX_ITERS = 1000

    LOADER_PARAMS = {
        'root_dir': DATASET_PATH, 'img_size': (256, 256), 'batch_size': 4
        }

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)
    
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Loading the Data Loader
    dataloader = load_dataset_ECOL_labeled(**LOADER_PARAMS)

    # Load the pretrained model
    model = models.vgg16(pretrained=True)
    
    # Replace the classification layer with the new one
    # Will output the prediction for only 1 class
    model.classifier[6] = nn.Linear(4096, 1)
    
    # Move model to GPU
    model = model.to(DEVICE)

    # dummy loop
    for data in dataloader:
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        labels = labels.unsqueeze(dim=1).to(torch.float32) 
        
        preds = model(inputs)

        loss = criterion(preds, labels)

        print(loss)

        break





