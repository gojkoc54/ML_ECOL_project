import os
PROJECT_ROOT = '/workspace/ml_ecol_project'
os.chdir(PROJECT_ROOT)

DATASET_PATH = os.path.join(PROJECT_ROOT, '../labeled_data')
CHECKPOINTS_PATH = os.path.join(PROJECT_ROOT, 'checkpoints')
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

from utils import *
from models import *

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

    # Loading the Data Loader
    dataloader = load_dataset_ECOL(**LOADER_PARAMS)

    # Load the pretrained model
    model = models.vgg16(pretrained=True)





