import torch
import torch.nn as nn

import os 
import numpy as np
import cv2
import rawpy 
import imageio 
from PIL import Image
from tqdm import tqdm

from utils import *


class AEContractingBlock(nn.Module):
    
    def __init__(
        self, in_channels, out_channels=None, 
        use_bn = True, use_dropout = False
        ):
        super(AEContractingBlock, self).__init__()
        
        if out_channels is None:
            out_channels = 2 * in_channels
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size = 4, stride = 2, padding = 1
            )
        
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.activation = nn.LeakyReLU(0.2)


    def forward(self, x):

        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.activation(out)

        return out



class AEExpandingBlock(nn.Module):

    def __init__(
        self, in_channels, out_channels=None, 
        use_bn = True, use_dropout = False
        ):
        super(AEExpandingBlock, self).__init__()

        if out_channels is None:
            out_channels = in_channels // 2
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size = 4, stride = 2, padding = 1
            )
        
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.activation = nn.ReLU() 
 

    def forward(self, x):
        
        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.activation(out)

        return out



class Encoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, depth):
        super(Encoder, self).__init__()

        self.depth = depth

        # Input -> hidden mapping
        self.set_hidden_channels = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1
            )

        # Contracting layers
        self.contracting_layers = nn.ModuleList()
        start_scale = int(hidden_channels / 2)
        contracting_scales = [2*i*start_scale for i in range(1, self.depth + 2)]
        for i in range(self.depth):
            curr_in_channels = contracting_scales[i]
            curr_out_channels = contracting_scales[i + 1]

            self.contracting_layers.append(
                AEContractingBlock(
                    curr_in_channels, curr_out_channels, use_dropout=True
                    )
                )
            

    def forward(self, x):
        out = self.set_hidden_channels(x)

        for i in range(self.depth):
            out = self.contracting_layers[i](out)

        return out



class Decoder(nn.Module):

    def __init__(self, hidden_channels, out_channels, depth):
        super(Decoder, self).__init__()

        self.depth = depth

        # Expanding layers
        self.expanding_layers = nn.ModuleList()
        start_scale = int(hidden_channels / 2)
        expanding_scales = [2*i*start_scale for i in range(1, self.depth + 2)]
        expanding_scales = expanding_scales[ : : -1]
        for i in range(self.depth):
            curr_in_channels = expanding_scales[i]
            curr_out_channels = expanding_scales[i + 1]
            
            self.expanding_layers.append(
                AEExpandingBlock(
                    curr_in_channels, curr_out_channels, use_dropout=True
                    )
                )

        # Hidden -> output mapping
        self.set_output_channels = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=1
            )
    

    def forward(self, x):
        out = x
        for i in range(self.depth):
            out = self.expanding_layers[i](out)

        out = self.set_output_channels(out)

        return out



class AutoencoderCNN(nn.Module):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        hidden_channels=64, 
        depth=4,
        checkpoint_dir='.'
        ):
        super(AutoencoderCNN, self).__init__()

        self.depth = depth
        self.checkpoint_dir = checkpoint_dir

        # Encoder
        self.encoder = Encoder(in_channels, hidden_channels, depth)
        # Decoder 
        self.decoder = Decoder(hidden_channels, out_channels, depth)

 
    def forward(self, x):
        encoding = self.encoder(x)
        out = self.decoder(encoding)
        
        return {'out': out, 'encoding': encoding}

    
    def train_epoch(self, loader, optimizer, epoch_idx, device, max_iters=None):
        self.train()
        batch_size = loader.batch_size
        running_loss = 0.0

        train_len = len(loader) if max_iters is None else max_iters

        progress_bar = tqdm(
            loader, total=train_len, desc=f'Epoch {epoch_idx}'
            )

        for batch_idx, data in enumerate(progress_bar):
            if max_iters is not None:
                if batch_idx == max_iters:
                    break

            data = data.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = self(data)['out']
            loss = nn.functional.mse_loss(output, data)
            loss.backward()
            optimizer.step()

            # log statistics
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            progress_bar.set_postfix(loss=avg_loss, stage="train")
        
        if max_iters is None:
            return running_loss / len(loader)
        else:
            return running_loss / max_iters 


    def evaluate(self, loader, device, epoch_idx=None, should_print=True):        
        self.eval()
        test_loss, correct = 0, 0

        if epoch_idx is None:
          description = 'Evaluating'
        else:
          description = f'Validation epoch {epoch_idx}'

        progress_bar = tqdm(loader, total=len(loader), desc=description)
        
        with torch.no_grad():
            for batch_idx, data in enumerate(progress_bar):
                data = data.to(device)

                output = self(data)['out']

                loss = nn.functional.cross_entropy(output, data)
                test_loss += loss.item()
                avg_loss = test_loss / (batch_idx + 1)

                progress_bar.set_postfix(loss=avg_loss, stage="validation")

        test_loss /= len(loader.dataset)

        if should_print:
          print(f'\nAvg. loss: {test_loss:.4f}')

        return test_loss


    def save_checkpoint(self, epoch=None, checkpoint_name=None):
        if checkpoint_name is None:
            checkpoint_name = 'latest_checkpoint.pt'
        save_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        torch.save({'model': self, 'epoch': str(epoch)}, save_path)


    def plot_progress(
        self, 
        loader, 
        num_batches, 
        save_dir, 
        device, 
        show_plot=False, 
        normalized=True
        ):

        self.eval()

        for i, data in enumerate(loader):
            if i == num_batches:
                break 
            data = data.to(device)
            pred = self(data)['out']

            in_out_batch = torch.cat([data, pred], dim=0)
            grid_img = torchvision.utils.make_grid(
                in_out_batch, nrow=loader.batch_size
                )
            
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'batch_{i}.png')

            figsize = (4 * loader.batch_size, 10)
            show_image(
                grid_img.cpu(), 
                normalized=normalized, 
                figsize=figsize, 
                save_path=save_path, 
                show_plot=show_plot
                )


    def fit(self, loaders, optimizer, epochs, device, max_iters=None):
        train_loader, val_loader = loaders

        train_losses, val_losses = [], []
        for epoch in range(epochs):
            # Training epoch
            running_loss = self.train_epoch(
                loader=train_loader, 
                optimizer=optimizer, 
                epoch_idx=epoch,
                device=device,
                max_iters=max_iters
                )
            train_losses.append(running_loss)

            # Validation
            if val_loader is not None:
                val_loss, _ = self.evaluate(
                    val_loader, device, should_print=False, epoch_idx=epoch
                    )
                val_losses.append(val_loss)

            # Save checkpoint
            self.save_checkpoint(epoch=epoch)

            # Plot current output images
            progress_plot_dir = os.path.join(
                self.checkpoint_dir, 'plots', f'progress_epoch_{epoch}'
                )
            os.makedirs(progress_plot_dir, exist_ok=True)

            self.plot_progress(train_loader, 5, progress_plot_dir, device)

        self.latest_train_losses = train_losses
        self.latest_val_losses = val_losses
