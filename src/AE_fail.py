import os
PROJECT_ROOT = '/workspace/ml_ecol_project'
os.chdir(PROJECT_ROOT)

DATASET_PATH = os.path.join(PROJECT_ROOT, '../dataset')
CHECKPOINTS_PATH = os.path.join(PROJECT_ROOT, 'checkpoints')
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

from utils import *
from models import *


if __name__ == '__main__':

    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    HIDDEN_CHANNELS = 2
    DEPTH = 2
    EPOCHS = 100
    LEARNING_RATE = 0.001
    MAX_ITERS = 1000

    AE_PARAMS = (
        IN_CHANNELS, OUT_CHANNELS, HIDDEN_CHANNELS, DEPTH, CHECKPOINTS_PATH
        )

    LOADER_PARAMS = {
        'root_dir': DATASET_PATH, 'img_size': (256, 256), 'batch_size': 4
        }

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    # Loading the Data Loader
    dataloader = load_dataset_ECOL(**LOADER_PARAMS)

    # Loading the AE model 
    model = AutoencoderCNN(*AE_PARAMS)
    model = model.to(DEVICE)

    # Loading the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    # Training
    model.fit([dataloader, None], optimizer, EPOCHS, DEVICE, MAX_ITERS)

