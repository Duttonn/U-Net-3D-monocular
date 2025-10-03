#!/usr/bin/env python
"""
Script d'aperçu rapide des prédictions pour validation visuelle
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet3d_lite import UNet3D_Lite
from src.datasets.simple_tartanair import SimpleTartanAirDataset


def load_config():
    """Configuration de base pour le test rapide"""
    return {
        "data_root": "data/tartanair-v2",
        "envs": ["ArchVizTinyHouseDay"],
        "difficulties": ["easy"],
        "cameras": ["lcam_front"],
        "modalities": ["image", "depth", "seg"],
        "T": 5,
        "size": [240, 320],
        "bs": 1,
        "num_classes": 6
    }


def quick_visualization():
    """Visualisation rapide d'une prédiction"""
    print("🚀 Quick Visualization - Test des prédictions")
    
    # Configuration
    cfg = load_config()
    
    # Device
    dev = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    
    # Dataset
    try:
        ds = SimpleTartanAirDataset(cfg)
        loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0)
        frames, depth_gt, seg_gt, valid = next(iter(loader))
        print(f"✅ Dataset chargé: frames {frames.shape}, depth {depth_gt.shape}")
    except Exception as e:
        print(f"❌ Erreur dataset: {e}")
        return
    
    # Modèle
    try:
        model = UNet3D_Lite(num_classes=cfg["num_classes"], pretrained=True).to(dev).eval()
        print(f"✅ Modèle chargé: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        return
    
    # Ajouter dimension batch pour l'inférence
    frames_batch = frames.unsqueeze(0).to(dev)  # [1, 5, 3, 240, 320]
    
    # Inférence
    with torch.no_grad():
        try:
            depth_pred, seg_logits = model(frames_batch)
            print(f"✅ Inférence: depth {depth_pred.shape}, seg {seg_logits.shape}")
        except Exception as e:
            print(f"❌ Erreur inférence: {e}")
            return
    
    # Visualisation
    try:
        # Extraire les données (enlever dimension batch)
        depth_pred_np = depth_pred[0, 0].cpu().numpy()  # [H, W]
        depth_gt_np = depth_gt[0].cpu().numpy()         # [H, W] 
        rgb_center = frames[2].permute(1, 2, 0).cpu().numpy()  # Frame centrale [H, W, 3]
        seg_pred = torch.argmax(seg_logits[0], dim=0).cpu().numpy()  # [H, W]
        seg_gt_np = seg_gt.cpu().numpy()  # [H, W]
        
        # Normaliser RGB
        rgb_center = (rgb_center * 255).astype(np.uint8)
        
        # Normaliser depth pour visualisation (0-255)
        if depth_pred_np.max() > 0:
            depth_pred_vis = (255 * (depth_pred_np / depth_pred_np.max())).astype(np.uint8)
        else:
            depth_pred_vis = np.zeros_like(depth_pred_np, dtype=np.uint8)
            
        if depth_gt_np.max() > 0:
            depth_gt_vis = (255 * (depth_gt_np / depth_gt_np.max())).astype(np.uint8)
        else:
            depth_gt_vis = np.zeros_like(depth_gt_np, dtype=np.uint8)
        
        # Coloriser segmentation
        seg_colors = np.array([
            [0, 0, 0],       # Classe 0: Noir
            [255, 0, 0],     # Classe 1: Rouge  
            [0, 255, 0],     # Classe 2: Vert
            [0, 0, 255],     # Classe 3: Bleu
            [255, 255, 0],   # Classe 4: Jaune
            [255, 0, 255],   # Classe 5: Magenta
        ])
        
        # Clamp des indices de segmentation pour éviter les erreurs
        seg_pred_clamped = np.clip(seg_pred, 0, len(seg_colors)-1)
        seg_gt_clamped = np.clip(seg_gt_np, 0, len(seg_colors)-1)
        
        seg_pred_vis = seg_colors[seg_pred_clamped].astype(np.uint8)
        seg_gt_vis = seg_colors[seg_gt_clamped].astype(np.uint8)
        
        # Sauvegarder les images
        cv2.imwrite("quick_rgb_center.png", cv2.cvtColor(rgb_center, cv2.COLOR_RGB2BGR))
        cv2.imwrite("quick_depth_pred.png", depth_pred_vis)
        cv2.imwrite("quick_depth_gt.png", depth_gt_vis)
        cv2.imwrite("quick_seg_pred.png", cv2.cvtColor(seg_pred_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite("quick_seg_gt.png", cv2.cvtColor(seg_gt_vis, cv2.COLOR_RGB2BGR))
        
        print("✅ Images sauvées:")
        print("  - quick_rgb_center.png (frame centrale)")
        print("  - quick_depth_pred.png (prédiction profondeur)")
        print("  - quick_depth_gt.png (ground truth profondeur)")
        print("  - quick_seg_pred.png (prédiction segmentation)")
        print("  - quick_seg_gt.png (ground truth segmentation)")
        
        # Stats rapides
        print(f"\n📊 Stats rapides:")
        print(f"  Depth pred: {depth_pred_np.min():.2f} - {depth_pred_np.max():.2f} m")
        print(f"  Depth GT:   {depth_gt_np.min():.2f} - {depth_gt_np.max():.2f} m")
        print(f"  Seg classes pred: {np.unique(seg_pred)}")
        print(f"  Seg classes GT:   {np.unique(seg_gt_np)}")
        
    except Exception as e:
        print(f"❌ Erreur visualisation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("🎉 Visualisation rapide terminée avec succès!")


if __name__ == "__main__":
    quick_visualization()