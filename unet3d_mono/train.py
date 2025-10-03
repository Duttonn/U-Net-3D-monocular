#!/usr/bin/env python3
"""
Training script for U-Net 3D monocular depth estimation with TartanAir
"""

import argparse
import os
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.datasets.simple_tartanair import SimpleTartanAirDataset
from src.models.unet3d_lite import UNet3D_Lite


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def depth_metrics(pred, gt, valid):
    """
    Calcule les métriques de profondeur standard
    Args:
        pred: [B, 1, H, W] prédictions de profondeur
        gt: [B, 1, H, W] ground truth
        valid: [B, 1, H, W] masque de pixels valides
    Returns:
        abs_rel, rmse, delta_1.25
    """
    eps = 1e-6
    p = pred.clamp(min=eps)
    g = gt.clamp(min=eps)
    v = (valid > 0.5).float()
    n = v.sum().clamp(min=1.0)
    
    # Absolute relative error
    abs_rel = (v * (p - g).abs() / g).sum() / n
    
    # RMSE
    rmse = torch.sqrt((v * (p - g) ** 2).sum() / n)
    
    # Delta < 1.25 (accuracy metric)
    ratio = torch.max(p / g, g / p)
    delta = (v.squeeze(1) * (ratio < 1.25)).sum() / n
    
    return abs_rel.item(), rmse.item(), delta.item()


def train_epoch(model, dataloader, criterion, optimizer, device, max_steps=50):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_abs_rel = 0.0
    total_rmse = 0.0
    total_delta = 0.0
    
    for step, (frames, depth_gt, seg_gt, valid_mask) in enumerate(dataloader):
        if step >= max_steps:
            break
            
        frames = frames.to(device)
        depth_gt = depth_gt.to(device)
        seg_gt = seg_gt.to(device)
        valid_mask = valid_mask.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        depth_pred, seg_logits = model(frames)
        
        # Loss combinée depth + segmentation
        depth_loss = torch.nn.functional.l1_loss(depth_pred * valid_mask, depth_gt * valid_mask)
        seg_loss = torch.nn.functional.cross_entropy(seg_logits, seg_gt.squeeze(1).long())
        
        loss = depth_loss + 0.1 * seg_loss  # Pondération seg plus faible
        
        loss.backward()
        optimizer.step()
        
        # Métriques
        abs_rel, rmse, delta = depth_metrics(depth_pred, depth_gt, valid_mask)
        
        total_loss += loss.item()
        total_abs_rel += abs_rel
        total_rmse += rmse
        total_delta += delta
        
        if step % 10 == 0:
            print(f'Step {step+1}/{max_steps}, Loss: {loss.item():.4f}, '
                  f'AbsRel: {abs_rel:.3f}, RMSE: {rmse:.3f}, δ<1.25: {delta:.3f}')
    
    n_steps = min(max_steps, step + 1)
    return (total_loss / n_steps, total_abs_rel / n_steps, 
            total_rmse / n_steps, total_delta / n_steps)


def main():
    parser = argparse.ArgumentParser(description='Train U-Net 3D monocular depth estimation')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f'Using device: {device}')
    
    # Create model - VRAI MODÈLE UNet3D_Lite
    model = UNet3D_Lite(num_classes=config.get("num_classes", 6), pretrained=True).to(device).train()
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Create dataset
    train_dataset = SimpleTartanAirDataset(config)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.get('bs', 4), 
        shuffle=True, 
        num_workers=config.get('num_workers', 0)  # Important: 0 pour éviter conflits
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('lr', 2.5e-4)
    )
    
    # Setup logging
    log_dir = Path(config.get('log_dir', 'runs/default'))
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Training loop
    epochs = config.get('epochs', 2)
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        
        # Train
        train_loss, train_abs_rel, train_rmse, train_delta = train_epoch(
            model, train_loader, None, optimizer, device
        )
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Metrics/AbsRel', train_abs_rel, epoch)
        writer.add_scalar('Metrics/RMSE', train_rmse, epoch)
        writer.add_scalar('Metrics/Delta1.25', train_delta, epoch)
        
        print(f'Epoch {epoch+1} Summary:')
        print(f'  Loss: {train_loss:.4f}')
        print(f'  AbsRel: {train_abs_rel:.3f}')
        print(f'  RMSE: {train_rmse:.3f}')
        print(f'  δ<1.25: {train_delta:.3f}')
        
        # Save checkpoint
        save_dir = Path(config.get('save_dir', 'ckpts/default'))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'config': config
        }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    writer.close()
    print(f'Training completed! Final metrics: Loss={train_loss:.4f}, AbsRel={train_abs_rel:.3f}')


if __name__ == '__main__':
    main()