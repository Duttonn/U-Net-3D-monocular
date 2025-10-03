#!/bin/bash
# Script de vérification post-nettoyage
# Valide que toutes les composantes du projet fonctionnent correctement

echo "🔍 Vérification post-nettoyage U-Net 3D Monocular"
echo "==============================================="

# Activer l'environnement si il existe
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Environnement virtuel activé"
else
    echo "⚠️  Environnement virtuel non trouvé - utilisez ./setup.sh d'abord"
    exit 1
fi

# Vérifier les imports Python
echo "🐍 Test des imports Python..."
python -c "
try:
    import torch
    import torchvision
    import cv2
    import yaml
    import numpy as np
    from src.models.unet3d_lite import UNet3D_Lite
    from src.datasets.simple_tartanair import SimpleTartanAirDataset
    from src.datasets.depth_utils import load_depth_tartan, seg_to_index
    print('✅ Tous les imports réussis')
except ImportError as e:
    print(f'❌ Erreur import: {e}')
    exit(1)
"

# Tester le modèle UNet3D_Lite
echo "🏗️  Test du modèle UNet3D_Lite..."
python -c "
import torch
from src.models.unet3d_lite import UNet3D_Lite

model = UNet3D_Lite(num_classes=6, pretrained=False)
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'✅ Modèle chargé: {params:.1f}M paramètres')

# Test forward pass
x = torch.randn(1, 5, 3, 240, 320)
with torch.no_grad():
    depth, seg = model(x)
print(f'✅ Forward pass: depth {depth.shape}, seg {seg.shape}')
"

# Vérifier la structure des données
echo "📊 Vérification des données..."
if [ -f "data/tartanair-v2/expanded_download_complete.txt" ]; then
    frames_count=$(find data/tartanair-v2 -name "*.png" | wc -l)
    echo "✅ Données trouvées: $frames_count fichiers PNG"
else
    echo "⚠️  Pas de données synthétiques - lancez ./setup.sh pour les générer"
fi

# Test du dataset loader
echo "🔄 Test du dataset loader..."
python -c "
from src.datasets.simple_tartanair import SimpleTartanAirDataset
import torch

cfg = {
    'data_root': 'data/tartanair-v2',
    'envs': ['ArchVizTinyHouseDay'],
    'difficulties': ['easy'],
    'cameras': ['lcam_front'],
    'T': 5,
    'size': [240, 320],
    'num_classes': 6
}

try:
    ds = SimpleTartanAirDataset(cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0)
    frames, depth, seg, valid = next(iter(loader))
    print(f'✅ Dataset loader: frames {frames.shape}, depth {depth.shape}, seg {seg.shape}')
except Exception as e:
    print(f'⚠️  Dataset loader: {e}')
"

# Vérifier les fichiers de configuration
echo "⚙️  Vérification des configurations..."
if [ -f "configs/indoor_lowcam.yaml" ]; then
    echo "✅ Configuration trouvée: configs/indoor_lowcam.yaml"
else
    echo "❌ Configuration manquante: configs/indoor_lowcam.yaml"
fi

# Test des scripts
echo "📜 Test des scripts..."
if [ -f "scripts/quick_vis.py" ]; then
    echo "✅ Script de visualisation: scripts/quick_vis.py"
else
    echo "❌ Script manquant: scripts/quick_vis.py"
fi

if [ -f "scripts/download_expanded_synthetic.py" ]; then
    echo "✅ Script de données: scripts/download_expanded_synthetic.py"
else
    echo "❌ Script manquant: scripts/download_expanded_synthetic.py"
fi

# Vérifier les permissions
echo "🔐 Vérification des permissions..."
if [ -x "setup.sh" ]; then
    echo "✅ setup.sh exécutable"
else
    echo "⚠️  setup.sh non exécutable - utilisez chmod +x setup.sh"
fi

if [ -x "run.sh" ]; then
    echo "✅ run.sh exécutable"  
else
    echo "⚠️  run.sh non exécutable - utilisez chmod +x run.sh"
fi

echo ""
echo "🎉 Vérification terminée !"
echo ""
echo "📋 Prochaines étapes recommandées:"
echo "  1. Lancer l'entraînement: ./run.sh"
echo "  2. Visualiser les prédictions: python scripts/quick_vis.py"
echo "  3. Surveiller avec TensorBoard: tensorboard --logdir runs/indoor_lowcam"