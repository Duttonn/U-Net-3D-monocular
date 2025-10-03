#!/bin/bash
# Script d'installation automatisé U-Net 3D Monocular Depth Estimation
# Optimisé pour macOS avec Apple Silicon (M1/M2/M3)

set -e  # Arrêt en cas d'erreur

echo "🚀 Installation U-Net 3D Monocular Depth Estimation"
echo "=================================================="

# Vérifier Python 3.8+
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 non trouvé. Veuillez installer Python 3.8+ d'abord."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python détecté: $PYTHON_VERSION"

# Créer environnement virtuel
if [ ! -d ".venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv .venv
else
    echo "✅ Environnement virtuel existant détecté"
fi

# Activer environnement
source .venv/bin/activate
echo "✅ Environnement virtuel activé"

# Mettre à jour pip
echo "🔄 Mise à jour de pip..."
pip install --upgrade pip

# Installer les dépendances
echo "📦 Installation des dépendances PyTorch..."
pip install torch torchvision torchaudio

echo "📦 Installation des autres dépendances..."
pip install -r requirements.txt

# Vérifier installation PyTorch
echo "🧪 Test de l'installation PyTorch..."
python3 -c "
import torch
print(f'✅ PyTorch {torch.__version__} installé')
if torch.backends.mps.is_available():
    print('✅ Support MPS (Apple Silicon) disponible')
else:
    print('ℹ️  MPS non disponible, utilisation CPU')
"

# Générer les données synthétiques par défaut
echo "📊 Génération des données synthétiques d'exemple..."
python scripts/download_expanded_synthetic.py --frames_per_env 50

# Test rapide du modèle
echo "🧪 Test du modèle UNet3D_Lite (14.1M params)..."
python -c "
from src.models.unet3d_lite import UNet3D_Lite
import torch
model = UNet3D_Lite(num_classes=6, pretrained=False)
print(f'✅ Modèle UNet3D_Lite chargé: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M paramètres')
x = torch.randn(1, 5, 3, 240, 320)
with torch.no_grad():
    depth, seg = model(x)
print(f'✅ Forward pass réussi: depth {depth.shape}, seg {seg.shape}')
"

echo ""
echo "🎉 Installation terminée avec succès!"
echo ""
echo "📋 Prochaines étapes:"
echo "  1. Activer l'environnement: source .venv/bin/activate"
echo "  2. Lancer l'entraînement: ./run.sh"
echo "  3. Visualisation rapide: python scripts/quick_vis.py"
echo ""
echo "📁 Structure du projet:"
echo "  - Modèle: src/models/unet3d_lite.py (14.1M params)"
echo "  - Config: configs/indoor_lowcam.yaml"
echo "  - Données: data/tartanair-v2/ (400 frames synthétiques)"
echo "  - Logs: runs/indoor_lowcam/"
echo "  - Checkpoints: ckpts/indoor_lowcam/"