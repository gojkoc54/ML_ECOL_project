import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

import os
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image 

class UnNormalize(object):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, tensor):
    """
    Args:
      tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
      Tensor: Normalized image.
    """
    for t, m, s in zip(tensor, self.mean, self.std):
      t.mul_(s).add_(m)
      # The normalize code -> t.sub_(m).div_(s)
    return tensor



class MetricTracker:
    def __init__(self):
        self.batches_cnt = 0
        self.total_loss = 0
        self.avg_loss = 0
        
        self.confusion_cnt = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        self.confusion_arrays = {'tp': [], 'tn': [], 'fp': [], 'fn': []}
        
        
    def update(self, loss, preds, labels):
        self.batches_cnt += 1
        
        self.total_loss += float(loss)
        self.avg_loss = self.total_loss / self.batches_cnt

        preds_sigmoid = torch.sigmoid(preds)
        preds_binary = preds_sigmoid > 0.5

        for i in range(len(labels)):        
            curr_key = ''
            if preds_binary[i] == 0 and labels[i] == 0:
                curr_key = 'tn'
            elif preds_binary[i] == 1 and labels[i] == 1:
                curr_key = 'tp'
            elif preds_binary[i] == 0 and labels[i] == 1:
                curr_key = 'fn'
            else:
                curr_key = 'fp'
                
            self.confusion_cnt[curr_key] += 1
            self.confusion_arrays[curr_key].append(float(preds_sigmoid[i]))
        
        
    def get_accuracy(self):
        confusion_cnt_sum = np.sum([cnt for _, cnt in self.confusion_cnt.items()])
        accuracy = self.confusion_cnt['tp'] + self.confusion_cnt['tn']
        accuracy /= confusion_cnt_sum
        
        return 100 * accuracy
    
    def get_precision(self):
        precision = self.confusion_cnt['tp'] / \
            (self.confusion_cnt['tp'] + self.confusion_cnt['fp'])
        
        return precision
    
    def get_recall(self):
        recall = self.confusion_cnt['tp'] / \
            (self.confusion_cnt['tp'] + self.confusion_cnt['fn'])
        
        return recall



def show_image(
    img, 
    figsize=(9, 6), 
    normalized=False, 
    save_path=None, 
    show_plot=True
    ):

    MEANS = [0.5] # (0.485, 0.456, 0.406)
    STDS = [0.25] # (0.229, 0.224, 0.225)
    unorm = UnNormalize(MEANS, STDS)
    
    img_to_show = img if not normalized else unorm(img)
    
    plt.figure(figsize=figsize)    
    plt.imshow(img_to_show.permute(1, 2, 0), vmin=0, vmax=255)
    if save_path is not None:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()



def check_balancing(dataloader):
    positives_cnt = 0
    for ind in dataloader.sampler.indices:
        positives_cnt += dataloader.dataset.samples[ind][1]

    positives_pctg = positives_cnt / len(dataloader.sampler.indices)
    print(f'Percentage of positive samples: {100 * positives_pctg:.2f} %')



def get_balanced_indices(dataset):

    positive_indices = [i for i in range(len(dataset.samples)) if dataset.samples[i][1] == 1]
    negative_indices = [i for i in range(len(dataset.samples)) if dataset.samples[i][1] == 0]
    negative_indices_balanced = np.random.choice(
        negative_indices, len(positive_indices), replace=False
        )
    
    return np.hstack([
        np.array(positive_indices), np.array(negative_indices_balanced)
        ])



class ECOLDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        super(ECOLDataset, self).__init__()

        self.root_dir = os.path.abspath(root_dir)
        self.subdirs = os.listdir(self.root_dir)
        self.subdirs.sort()

        self.dir_label_map = {}
        for i, subdir in enumerate(self.subdirs):
            self.dir_label_map[subdir] = i

        # Iterate through the subdirs and collect (path_to_img, label) pairs
        self.paths_and_labels = []
        for subdir in self.subdirs:
            subdir_path = os.listdir(os.path.join(self.root_dir, subdir))
            for img_name in subdir_path:
                img_path = os.path.join(subdir_path, img_name)
                self.paths_and_labels.append((img_path, self.dir_label_map[subdir]))
        
        self.len = len(self.paths_and_labels)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([ 
                transforms.ToTensor()
                ])

    
    def __len__(self):
        return self.len 

    
    def __getitem__(self, idx):

        img_path, label = self.paths_and_labels[idx]

        with open(img_path, 'rb') as img_file:
            img = Image.open(img_file)
        img = img.convert('RGB')

        return img, label, img_path



def load_dataset_ECOL_labeled(
    root_dir, 
    img_size, 
    batch_size, 
    num_workers=0, 
    shuffle=True,
    balance=True,
    transform=None,
    dataset_size=None,
    test_size=0.2
    ):

    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            # transforms.Grayscale(num_output_channels=3),
            transforms.Normalize(mean=MEANS, std=STDS),
            ])

    # dataset = ImageFolder(root_dir, transform=transform) 
    dataset = ECOLDataset(root_dir, transform=transform)
    
    if dataset_size is None:
        dataset_size = len(dataset)
        
    if balance:
        dataset_indices = get_balanced_indices(dataset)
        
        if len(dataset_indices) < dataset_size:
            dataset_size = len(dataset_indices)
    else:
        dataset_indices = list(range(len(dataset)))
    
    
    if shuffle:
        dataset_indices = np.random.choice(
            dataset_indices, dataset_size, replace=False
            )
    else:
        dataset_indices = dataset_indices[ : dataset_size]
    
    val_split_index = int(np.floor(test_size * dataset_size))

    train_indices, val_indices = \
        dataset_indices[val_split_index:], dataset_indices[:val_split_index]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset=dataset, 
        shuffle=False, 
        batch_size=batch_size, 
        sampler=train_sampler
        )
    
    val_loader = DataLoader(
        dataset=dataset, 
        shuffle=False, 
        batch_size=batch_size, 
        sampler=val_sampler
        )

    return train_loader, val_loader



def show_batch_as_grid(input_batch, save_path=None):
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]
    unorm = UnNormalize(MEANS, STDS)

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
    is_true_positive = (
        (preds_binary == 1) * (labels == 1)
        ).to(torch.bool).squeeze()

    if is_true_positive.sum() == 0:
        return

    true_positives = inputs[is_true_positive]

    show_batch_as_grid(true_positives, save_path)


