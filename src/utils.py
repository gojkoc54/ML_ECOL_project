from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

import os
import matplotlib.pyplot as plt 
import cv2 


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


class ECOLDataset(Dataset):

    def __init__(self, root_dir, transform=None, img_size=(512, 512)):
        super(ECOLDataset, self).__init__()

        self.root_dir = os.path.abspath(root_dir)
        self.MEANS = (0.5) # (0.485, 0.456, 0.406)
        self.STDS = (0.25) # (0.229, 0.224, 0.225)
        self.img_size = img_size

        # Iterate through the dataset iterator and count the elements
        dataset_iterator = os.scandir(root_dir)
        i = -1
        for i, _ in enumerate(dataset_iterator):
            pass

        self.len = i + 1

        # Torchvision transformation to be applied to each image
        if transform is not None:
            self.transform = transform 
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.img_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize(mean=self.MEANS, std=self.STDS),
                ])
            

    def __len__(self):
        # Has to exist in order to inherit the parent class properly!
        return self.len


    def __getitem__(self, idx):

        dataset_iterator = os.scandir(self.root_dir)
        img_path = None
        for i, curr_img_name in enumerate(dataset_iterator):
            if i == idx:
                img_path = os.path.join(self.root_dir, curr_img_name.name)
                break

        image = cv2.imread(img_path)
        
        if image is None:
            print(img_path)
            return None 

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed_image = self.transform(image)

        return transformed_image



def load_dataset_ECOL(
    root_dir, 
    img_size, 
    batch_size, 
    num_workers=0, 
    shuffle=True,
    transform=None,
    ):

    dataset = ECOLDataset(root_dir, transform, img_size) 

    # Create DataLoader object for the dataset  
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        ) 

    return loader



def load_dataset_ECOL_labeled(
    root_dir, 
    img_size, 
    batch_size, 
    num_workers=0, 
    shuffle=True,
    transform=None,
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

    dataset = ImageFolder(root_dir, transform=transform) 

    # Create DataLoader object for the dataset  
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        ) 

    return loader

