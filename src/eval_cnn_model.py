from PIL.Image import init
import torch.nn as nn
import os
from utils import *
from training_utils import *

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--bs', '--batch-size', default=32, type=int)
parser.add_argument('--img-size', default=(256, 256), type=tuple)
parser.add_argument('--model', default='vgg16', type=str)
parser.add_argument('--balance', default=1, type=int)

parser.add_argument('--root', default='/workspace/ml_ecol_project', type=str)
parser.add_argument('--test-dir', default='../labeled_data_test', type=str)
parser.add_argument('--cp-path', default='checkpoints', type=str)

args = parser.parse_args()

MODEL_CLASS_DICT = {
    'alexnet': 'alexnet', 'vgg16': 'vgg', 
    'resnet18': 'resnet', 'densenet121': 'densenet'
    }

if __name__ == '__main__':

    # Define the paths
    os.chdir(args.root)
    TEST_DATA_PATH = os.path.join(args.root, args.test_dir)

    TEST_LOADER_PARAMS = {
        'root_dir': TEST_DATA_PATH, 'img_size': args.img_size, 
        'batch_size': args.bs, 'test_size': 0, 'balance': bool(args.balance)
        }

    # Define the destination device for training
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice being used: {DEVICE}\n')
    
    # Loading the Data Loader
    test_loader, _ = load_dataset_ECOL_labeled(**TEST_LOADER_PARAMS)

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Load the desired checkpoint 
    CHECKPOINTS_PATH = os.path.join(args.root, args.cp_path, args.model)
    checkpoint_load_path = os.path.join(
        CHECKPOINTS_PATH, 
        f'{MODEL_CLASS_DICT[args.model]}_checkpoint.pt'
        )

    checkpoint = torch.load(checkpoint_load_path)
    print(f'Loaded checkpoint from path {checkpoint_load_path}\n')

    # Load the pretrained model from the checkpoint
    model = initialize_model(args.model, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move model to GPU
    model = model.to(DEVICE)

    # Evaluate the model on the test set
    test_metrics = evaluate(
        model, test_loader, criterion, DEVICE, title='TESTING', save_paths=True
        )

    # Save the metrics
    metrics_save_path = os.path.join(
        CHECKPOINTS_PATH, 
        f'{type(model).__name__.lower()}_test_metrics.pt'
        )    
    torch.save(test_metrics, metrics_save_path)
    print(f'\nMetrics saved to {metrics_save_path}\n')




