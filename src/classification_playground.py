import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger()
old_level = logger.level
logger.setLevel(100)

import os
PROJECT_ROOT = '/workspace/ml_ecol_project'
os.chdir(PROJECT_ROOT)

DATASET_PATH = os.path.join(PROJECT_ROOT, '../labeled_data')
CHECKPOINTS_PATH = os.path.join(PROJECT_ROOT, 'checkpoints')
PLOTS_PATH = os.path.join(CHECKPOINTS_PATH, 'plots')
os.makedirs(PLOTS_PATH, exist_ok=True)

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



MEANS = [0.485, 0.456, 0.406]
STDS = [0.229, 0.224, 0.225]
unorm = UnNormalize(MEANS, STDS)


def show_batch_as_grid(input_batch, save_path=None):
    
    grid_img = torchvision.utils.make_grid(
        unorm(input_batch).cpu()
        )

    plt.figure(figsize=(6 * input_batch.shape[0], 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

    
def visualize_false_negatives(inputs, labels, preds, save_path=None):
    preds_binary = preds > 0.5
    is_false_negative = ((preds_binary == 0) * (labels == 1)).to(torch.bool).squeeze()
    
    if is_false_negative.sum() == 0:
        return 

    false_negatives = inputs[is_false_negative] 
    
    show_batch_as_grid(false_negatives, save_path)


def visualize_true_positives(inputs, labels, preds, save_path=None):
    preds_binary = preds > 0.5
    is_true_positive = ((preds_binary == 1) * (labels == 1)).to(torch.bool).squeeze()

    if is_true_positive.sum() == 0:
        return

    true_positives = inputs[is_true_positive]

    show_batch_as_grid(true_positives, save_path)




if __name__ == '__main__':

    EPOCHS = 100
    LEARNING_RATE = 0.0005  # 0.001
    MAX_ITERS = 1000

    LOADER_PARAMS = {
        'root_dir': DATASET_PATH, 'img_size': (256, 256), 'batch_size': 32
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
        
        os.makedirs(os.path.join(PLOTS_PATH, f'epoch_{epoch}'), exist_ok=True)

        running_loss = 0
        running_hits = 0
        num_positives = 0
        total_positives = 0
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
            total_positives += labels.sum()
            
            if preds.max() > max_pred:
                max_pred = float(preds.max())

            if i % 10 == 0:
                print(f'Epoch {epoch}, Batch {i}/{total_batch_num}: {float(loss)}')
            
            if i % 30 == 0:
                save_path = os.path.join(PLOTS_PATH, f'epoch_{epoch}', f'false_negative_batch_{i}.png')
                visualize_false_negatives(inputs, labels, preds, save_path)
                
                save_path = os.path.join(PLOTS_PATH, f'epoch_{epoch}', f'true_positive_batch_{i}.png')
                visualize_true_positives(inputs, labels, preds, save_path)

        losses.append(running_loss / total_batch_num) 
        accuracies.append(running_hits / len(dataloader.dataset))
        
        print(f'\n=== Epoch {epoch} ===')
        print(f'Avg loss = {losses[-1]}')
        print(f'ACC = {accuracies[-1]}')
        print(f'positives  = {100 * num_positives / len(dataloader.dataset)}% ; {num_positives} / {total_positives}')
        print(f'max pred = {max_pred}\n')


