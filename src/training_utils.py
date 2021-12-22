import torch
import torch.nn as nn

from utils import *


def train_epoch(model, dataloader, optimizer, criterion, device, epoch_idx=0):
    
    # Put the model in training mode
    model.train()
    
    # Initialize the metrics
    metric_tracker = MetricTracker()

    if 'sampler' in dir(dataloader):
        total_batch_num = \
            len(dataloader.sampler.indices) // dataloader.batch_size
    else:
        total_batch_num = len(dataloader.dataset) // dataloader.batch_size
    
    for i, data in enumerate(dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        labels = labels.unsqueeze(dim=1).to(torch.float32) 
        
        # Reset the optimizer gradient
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(inputs)
            
        loss = criterion(preds, labels)
        
        # Backwards pass and parameter update
        loss.backward()
        optimizer.step()
        
        # Update the metrics
        metric_tracker.update(float(loss), preds, labels)
        
        if i % 10 == 0:
            msg = f'Train epoch {epoch_idx},'
            msg += f' Batch {i}/{total_batch_num}: {float(loss)}'
            print(msg)
    
    print(f'\n=== TRAIN - Epoch {epoch_idx} ===')
    print(f'Avg loss = {metric_tracker.avg_loss}')
    print(f'Accuracy = {metric_tracker.get_accuracy()}')
    print(f'Precision = {metric_tracker.get_precision()}')
    print(f'Recall = {metric_tracker.get_recall()}')
    print()
    
    return metric_tracker


def evaluate(model, dataloader, criterion, device):
    
    # Put the model in eval mode
    model.eval()
    
    # Initialize the metrics
    metric_tracker = MetricTracker()

    if 'sampler' in dir(dataloader):
        total_batch_num = \
            len(dataloader.sampler.indices) // dataloader.batch_size
    else:
        total_batch_num = len(dataloader.dataset) // dataloader.batch_size
    
    for i, data in enumerate(dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        labels = labels.unsqueeze(dim=1).to(torch.float32) 
        
        # Forward pass
        preds = model(inputs)
            
        loss = criterion(preds, labels)
        
        # Update the metrics
        metric_tracker.update(float(loss), preds, labels)
        
        if i % 10 == 0:
            print(f'Validation - Batch {i}/{total_batch_num}: {float(loss)}')
    
    print(f'\n=== VALIDATION ===')
    print(f'Avg loss = {metric_tracker.avg_loss}')
    print(f'Accuracy = {metric_tracker.get_accuracy()}')
    print(f'Precision = {metric_tracker.get_precision()}')
    print(f'Recall = {metric_tracker.get_recall()}')
    print()
    
    return metric_tracker



def fit(model, loaders, optimizer, criterion, device, epochs=10):
    
    train_loader, val_loader = loaders
    metrics_per_epoch = {}

    for epoch in range(epochs):

        metrics_per_epoch[epoch] = {}
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
            )
        metrics_per_epoch[epoch]['train'] = train_metrics
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        metrics_per_epoch[epoch]['val'] = val_metrics

    return model
