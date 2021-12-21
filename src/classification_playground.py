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


# TODO:
#   - train/val split
#   - checkpointing 
#   - play with different architectures
#   - enrich the dataset
#   - balance the dataset
#   - !!! inspect the images that are positive but the model predicts them 
#         as negative; after X epochs when it saturates


if __name__ == '__main__':

    EPOCHS = 100
    LEARNING_RATE = 0.001
    MAX_ITERS = 1000

    LOADER_PARAMS = {
        'root_dir': DATASET_PATH, 'img_size': (256, 256), 'batch_size': 64
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

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
   
    losses = []
    accuracies = []
    
    total_batch_num = len(dataloader.dataset) // dataloader.batch_size
    # Training loop
    for epoch in range(30):
        running_loss = 0
        running_hits = 0
        num_positives = 0
        max_pred = 0
        for i, data in enumerate(dataloader):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            labels = labels.unsqueeze(dim=1).to(torch.float32) 
        
            optimizer.zero_grad()

            preds = model(inputs)

            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()
            
            running_loss += float(loss)

            num_hits = (labels == (preds > 0.5)).sum()
            running_hits += num_hits
            
            num_positives += (preds > 0.5).sum()
            if preds.max() > max_pred:
                max_pred = float(preds.max())

            if i % 10 == 0:
                print(f'Epoch {epoch}, Batch {i}/{total_batch_num}: {float(loss)}')

        losses.append(running_loss / total_batch_num) 
        accuracies.append(running_hits / len(dataloader.dataset))
        
        print(f'\n=== Epoch {epoch} ===')
        print(f'Avg loss = {losses[-1]}')
        print(f'ACC = {accuracies[-1]}')
        print(f'positives % = {100 * num_positives / len(dataloader.dataset)}%')
        print(f'max pred = {max_pred}\n')


