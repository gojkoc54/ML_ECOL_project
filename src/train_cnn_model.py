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
parser.add_argument('--train-dir', default='../labeled_data', type=str)
parser.add_argument('--test-dir', default='../labeled_data_test', type=str)
parser.add_argument('--cp-path', default='checkpoints', type=str)
parser.add_argument('--plots-path', default='plots', type=str)

args = parser.parse_args()



if __name__ == '__main__':

    # Define the paths
    os.chdir(args.root)
    TRAIN_DATA_PATH = os.path.join(args.root, args.train_dir)
    TEST_DATA_PATH = os.path.join(args.root, args.test_dir)

    # Hyperparameters for the DataLoader
    TRAIN_LOADER_PARAMS = {
        'root_dir': TRAIN_DATA_PATH, 'img_size': args.img_size, 
        'batch_size': args.bs, 'test_size': 0.2, 'balance': True
        }

    TEST_LOADER_PARAMS = {
        'root_dir': TEST_DATA_PATH, 'img_size': args.img_size, 
        'batch_size': args.bs, 'test_size': 0, 'balance': False
        }

    # Define the destination device for training
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice being used: {DEVICE}\n')
    
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Loading the Data Loaders
    train_loader, val_loader = load_dataset_ECOL_labeled(**TRAIN_LOADER_PARAMS)
    test_loader, _ = load_dataset_ECOL_labeled(**TEST_LOADER_PARAMS)

    loaders = [train_loader, val_loader, test_loader]

    # Load the pretrained model
    # Replace the classification layer with the new one !!!
    # The new output layer will have a prediction for only 1 class.
    model = initialize_model(args.model, pretrained=args.pretrained)
    print(f'Loaded pre-trained model {type(model).__name__}\n')

    # If the model was properly loaded, create the checkpoint directory
    CHECKPOINTS_PATH = os.path.join(args.root, args.cp_path, args.model)
    PLOTS_PATH = os.path.join(CHECKPOINTS_PATH, args.plots_path)
    os.makedirs(PLOTS_PATH, exist_ok=True)

    # Move model to GPU
    model = model.to(DEVICE)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
   
    # Start training  
    paths_dict = {'cps_path': CHECKPOINTS_PATH, 'plots_path': PLOTS_PATH}
    model = fit(
        model, loaders, optimizer, criterion, DEVICE, paths_dict, args.epochs
        )

