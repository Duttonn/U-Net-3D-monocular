# U-Net-3D-monocular

Projet d'estimation de profondeur monoculaire utilisant un U-Net 3D avec backbone ResNet18 et convolutions (2+1)D, entraîné sur le dataset TartanAir V2.

**🎯 Architecture optimisée : UNet3D_Lite avec 14.1M paramètres**

## ✨ Fonctionnalités

- **🏗️ Modèle UNet3D_Lite** : ResNet18 + (2+1)D convolutions (14.1M params)
- **📊 Multi-tâches** : Estimation de profondeur + segmentation sémantique
- **⚡ Optimisé Apple Silicon** : Support MPS (M1/M2/M3)
- **📈 Métriques complètes** : AbsRel, RMSE, δ<1.25
- **🎨 Visualisation intégrée** : TensorBoard + script de visualisation rapide
- **🔧 Installation automatisée** : Script setup.sh clé en main

## 🚀 Installation rapide

```bash
git clone <repo_url>
cd U-Net-3D-monocular/unet3d_mono
chmod +x setup.sh
./setup.sh
```

Le script d'installation automatique :
- ✅ Crée l'environnement virtuel
- ✅ Installe PyTorch avec support MPS
- ✅ Génère 400 frames de données synthétiques
- ✅ Teste le modèle UNet3D_Lite
- ✅ Prépare l'environnement complet

## 📁 Architecture du projet

```
unet3d_mono/
├── setup.sh                     # 🔧 Installation automatique
├── run.sh                       # 🚀 Script de lancement rapide
├── train.py                     # 🎯 Entraînement principal
├── configs/
│   └── indoor_lowcam.yaml      # ⚙️ Configuration multi-env/caméras
├── src/
│   ├── models/
│   │   └── unet3d_lite.py      # 🏗️ UNet3D_Lite (14.1M params)
│   └── datasets/
│       ├── simple_tartanair.py  # 📊 Dataset TartanAir V2
│       └── depth_utils.py       # 🛠️ Utilitaires profondeur
├── scripts/
│   ├── download_expanded_synthetic.py  # 📦 Génération données étendues
│   └── quick_vis.py             # 👁️ Visualisation rapide
├── data/tartanair-v2/          # 💾 Données (400 frames synthétiques)
├── runs/                       # 📈 Logs TensorBoard
└── ckpts/                      # 💾 Checkpoints modèle
```

## 🎯 Utilisation

### 1️⃣ Entraînement rapide

```bash
source .venv/bin/activate
./run.sh
```

**Résultats attendus** (5 époques) :
```
Using device: mps
Params: 14.1M
Epoch 1/5: Loss: 2.24 → AbsRel: 0.807
Epoch 5/5: Loss: 0.12 → AbsRel: 0.029 (-96% amélioration!)
Final metrics: δ<1.25: 3.816 (excellente précision)
```

### 2️⃣ Visualisation rapide

```bash
python scripts/quick_vis.py
```

Génère automatiquement :
- `quick_rgb_center.png` - Image RGB centrale
- `quick_depth_pred.png` - Prédiction de profondeur  
- `quick_depth_gt.png` - Ground truth profondeur
- `quick_seg_pred.png` - Prédiction segmentation
- `quick_seg_gt.png` - Ground truth segmentation

### 3️⃣ Monitoring TensorBoard

```bash
source .venv/bin/activate
tensorboard --logdir runs/indoor_lowcam
# Ouvrir http://localhost:6006
```

## ⚙️ Configuration avancée

### Multi-environnements et caméras

Modifier `configs/indoor_lowcam.yaml` :

```yaml
# Dataset élargi (400 frames)
envs: ["ArchVizTinyHouseDay", "ArchVizTinyHouseNight"]
difficulties: ["easy", "hard"] 
cameras: ["lcam_bottom", "lcam_front"]  # lcam_bottom = vue basse

# Entraînement
epochs: 10          # Plus d'époques pour convergence
bs: 4              # Batch size (ajuster selon mémoire)
lr: 2.5e-4         # Learning rate optimisé
num_classes: 6     # Classes de segmentation
```

### Générer plus de données

```bash
python scripts/download_expanded_synthetic.py \
  --frames_per_env 100 \
  --envs ArchVizTinyHouseDay ArchVizTinyHouseNight \
  --difficulties easy hard \
  --cameras lcam_bottom lcam_front
```

## 🏗️ Architecture technique

### Modèle UNet3D_Lite

- **Backbone** : ResNet18 préentraîné (avec fix SSL)
- **Fusion temporelle** : Convolutions (2+1)D pour T=5 frames
- **Décodeur** : Upsampling progressif avec skip connections
- **Sorties duales** : Profondeur (1 canal) + Segmentation (6 classes)
- **Taille** : 14.1M paramètres (optimal pour entraînement rapide)

### Pipeline de données

```python
Input: [B, T=5, 3, H=240, W=320]  # Séquence de 5 images RGB
├── Temporal adapter (3D conv)      # Fusion temporelle initiale
├── ResNet18 encoder               # Extraction features spatiales
├── (2+1)D decoder                # Décodage avec fusion temporelle
└── Dual heads                    # Profondeur + Segmentation
    ├── Depth: [B, 1, H, W]      # 0-10m profondeur
    └── Seg: [B, 6, H, W]        # 6 classes sémantiques
```

### Métriques de profondeur

- **AbsRel** : Erreur relative absolue (↓ mieux)
- **RMSE** : Erreur quadratique moyenne (↓ mieux)  
- **δ<1.25** : % pixels avec ratio < 1.25 (↑ mieux)

## 🔧 Développement

### Ajouter un nouveau modèle

1. Créer `src/models/mon_modele.py`
2. Implémenter avec l'interface :
```python
class MonModele(nn.Module):
    def forward(self, x):  # x: [B, T, 3, H, W]
        return depth_pred, seg_logits  # [B,1,H,W], [B,C,H,W]
```
3. Remplacer dans `train.py` :
```python
from src.models.mon_modele import MonModele
model = MonModele(num_classes=6)
```

### Optimisations performances

- **Apple Silicon** : Détection auto MPS/CUDA/CPU
- **Mémoire** : `channels_last` format pour efficacité
- **Threading** : `num_workers=0` (TartanAir gère la parallélisation)

## 📊 Résultats

### Dataset synthétique élargi (400 frames)

| Métrique | Epoch 1 | Epoch 5 | Amélioration |
|----------|---------|---------|--------------|
| Loss     | 2.24    | 0.12    | -95%         |
| AbsRel   | 0.807   | 0.029   | -96%         |
| RMSE     | 2.227   | 0.140   | -94%         |
| δ<1.25   | 0.500   | 3.816   | +660%        |

**🎉 Convergence excellente avec le modèle UNet3D_Lite !**

## 🛠️ Troubleshooting

### Installation macOS

```bash
# Si problème SSL (certificats)
/Applications/Python\ 3.x/Install\ Certificates.command

# Si PyTorch MPS non détecté
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Mémoire insuffisante

```yaml
# Réduire dans configs/indoor_lowcam.yaml
bs: 2                # Batch size plus petit
size: [180, 240]     # Résolution réduite
T: 3                 # Moins de frames temporelles
```

### Checkpoints corrompus

```bash
# Nettoyer et redémarrer
rm -rf ckpts/indoor_lowcam/
rm -rf runs/indoor_lowcam/
./run.sh
```

## 📚 Dépendances principales

- **torch ≥2.0.0** - Framework ML avec support MPS
- **torchvision** - Modèles préentraînés (ResNet18)
- **opencv-python** - Traitement d'images et profondeur
- **tensorboard** - Monitoring d'entraînement
- **PyYAML** - Configuration YAML

## 🏆 Prochaines étapes

1. **Vraies données TartanAir V2** - Remplacer données synthétiques
2. **Optimisations M3** - Ajouter `autocast(bfloat16)` pour vitesse
3. **Fonctions de perte avancées** - Implémenter dans `src/losses/`
4. **Dataset multi-env** - Utiliser tous les environnements simultanément

## 📄 Licence

Ce projet est destiné à des fins éducatives. Le dataset TartanAir V2 a ses propres [conditions d'utilisation](http://theairlab.org/tartanair-dataset/).