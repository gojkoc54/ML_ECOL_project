import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(100)
import os

from utils import *
from training_utils import *

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--bs', '--batch-size', default=32, type=int)
parser.add_argument('--img-size', default=(256, 256), type=tuple)
parser.add_argument('--model', default='vgg16', type=str)
parser.add_argument('--pretrained', default=1, type=int)

parser.add_argument('--root', default='/workspace/ml_ecol_project', type=str)
parser.add_argument('--data-dir', default='../labeled_data', type=str)
parser.add_argument('--cp-path', default='checkpoints', type=str)
parser.add_argument('--plots-path', default='plots', type=str)

args = parser.parse_args()


# TODO:
#   - checkpointing 
#   - early stopping - save the model but continue learning
#   - logging and pickle-ing

#   - show histograms and learning curves

#   - !!! inspect the images that are positive but the model predicts them 
#         as negative; use the whole balanced dataset
#   - implement a dataset that works same as ImageFolder, BUT also returns
#         the path of the images ==>> save them when it's misclassified 

#   - test on unbalanced dataset BUT without the samples from training

# - ! add conda yaml file and check if it works



if __name__ == '__main__':

    # Define the paths
    os.chdir(args.root)
    DATASET_PATH = os.path.join(args.root, args.data_dir)

    # Hyperparameters for the DataLoader
    LOADER_PARAMS = {
        'root_dir': DATASET_PATH, 'img_size': args.img_size, 
        'batch_size': args.bs, 'test_size': 0.2, 'balance': True
        }

    # Define the destination device for training
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)
    
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Loading the Data Loader
    train_loader, val_loader = load_dataset_ECOL_labeled(**LOADER_PARAMS)
    loaders = [train_loader, val_loader]

    # Load the pretrained model
    # Replace the classification layer with the new one !!!
    # The new output layer will have a prediction for only 1 class.
    model = initialize_model(args.model, pretrained=args.pretrained)
    print(f'Loaded pre-trained model {type(model).__name__}')

    # If the model was properly loaded, create the checkpoint directory
    CHECKPOINTS_PATH = os.path.join(args.root, args.cp_path, args.model)
    PLOTS_PATH = os.path.join(CHECKPOINTS_PATH, args.plots_path)
    os.makedirs(PLOTS_PATH, exist_ok=True)

    # Move model to GPU
    model = model.to(DEVICE)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
   
    # Start training  
    model = fit(model, loaders, optimizer, criterion, DEVICE)

