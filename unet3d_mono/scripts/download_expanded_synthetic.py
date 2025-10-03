#!/usr/bin/env python
"""
Script de téléchargement élargi TartanAir V2 pour plus de diversité
Support multi-environnements et caméras basses
"""
import numpy as np
import cv2
from pathlib import Path
import argparse
import itertools

def create_expanded_synthetic_data(root, envs, difficulties, cameras, num_frames_per_env=200):
    """
    Génère des données synthétiques plus diversifiées pour simuler 
    plusieurs environnements et caméras
    """
    print(f"🔄 Génération de données synthétiques étendues...")
    print(f"Environnements: {envs}")
    print(f"Difficultés: {difficulties}")  
    print(f"Caméras: {cameras}")
    print(f"Frames par env: {num_frames_per_env}")
    
    root = Path(root)
    total_generated = 0
    
    for env, difficulty, camera in itertools.product(envs, difficulties, cameras):
        env_path = root / env / difficulty.title() / "P000"
        
        image_path = env_path / f"image_{camera}"
        depth_path = env_path / f"depth_{camera}"
        seg_path = env_path / f"seg_{camera}"
        
        for path in [image_path, depth_path, seg_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Génération: {env}/{difficulty}/{camera}")
        
        # Paramètres variables selon l'environnement
        if "Night" in env:
            brightness_factor = 0.3  # Plus sombre la nuit
            noise_level = 0.2
        else:
            brightness_factor = 1.0
            noise_level = 0.1
            
        if "hard" in difficulty.lower():
            motion_intensity = 2.0  # Mouvements plus rapides
            object_count = 4
        else:
            motion_intensity = 1.0
            object_count = 2
            
        if "bottom" in camera:
            perspective_offset = -0.3  # Caméra plus basse
            fov_factor = 1.2
        else:
            perspective_offset = 0.0
            fov_factor = 1.0
        
        for i in range(num_frames_per_env):
            frame_name = f"{i:06d}.png"
            
            # Image RGB avec variations environnementales
            img = create_env_image(240, 320, i, env, brightness_factor, 
                                 motion_intensity, object_count, perspective_offset)
            cv2.imwrite(str(image_path / frame_name), 
                       cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Depth map avec variations de caméra
            depth = create_env_depth(240, 320, i, fov_factor, 
                                   motion_intensity, perspective_offset)
            cv2.imwrite(str(depth_path / frame_name), depth)
            
            # Segmentation avec plus de classes
            seg = create_env_seg(240, 320, i, object_count, env)
            cv2.imwrite(str(seg_path / frame_name), seg)
            
            total_generated += 1
            
            if (i + 1) % 50 == 0:
                print(f"  Généré {i + 1}/{num_frames_per_env} frames")
    
    # Fichier indicateur
    (root / "expanded_download_complete.txt").write_text(
        f"Expanded synthetic data: {total_generated} frames total\n"
        f"Envs: {envs}\nDifficulties: {difficulties}\nCameras: {cameras}"
    )
    
    print(f"\n✅ Génération terminée! Total: {total_generated} frames")
    return total_generated


def create_env_image(height, width, frame_id, env_name, brightness, motion_intensity, object_count, perspective_offset):
    """Crée une image RGB avec variations environnementales"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Gradient de fond selon environnement
    if "Night" in env_name:
        # Couleurs nuit: bleu foncé + violet
        for y in range(height):
            img[y, :, 0] = int(50 * brightness * y / height)  # Rouge faible
            img[y, :, 1] = int(30 * brightness * y / height)  # Vert faible  
            img[y, :, 2] = int(100 * brightness + 100 * y / height)  # Bleu dominant
    else:
        # Couleurs jour: gradient chaud
        for y in range(height):
            img[y, :, 0] = int(255 * brightness * y / height)  
            img[y, :, 1] = int(200 * brightness * (1 - 0.3 * y / height))
            img[y, :, 2] = int(150 * brightness * (1 - 0.5 * y / height))
    
    # Objets mobiles avec intensité variable
    for obj_i in range(object_count):
        # Position avec mouvement et perspective
        base_x = width * (0.2 + 0.6 * obj_i / max(1, object_count - 1))
        base_y = height * (0.4 + perspective_offset)
        
        center_x = int(base_x + motion_intensity * 30 * np.sin(frame_id * 0.05 + obj_i))
        center_y = int(base_y + motion_intensity * 20 * np.cos(frame_id * 0.07 + obj_i))
        
        # Couleur selon type d'objet
        if obj_i == 0:
            color = (255, 255, 255)  # Objet principal blanc
            radius = 25
        elif obj_i == 1:
            color = (0, 255, 255)    # Objet secondaire cyan
            radius = 20
        else:
            color = (255, 0, 255)    # Autres objets magenta
            radius = 15
        
        cv2.circle(img, (center_x, center_y), radius, color, -1)
    
    # Appliquer brightness globale
    img = (img * brightness).astype(np.uint8)
    
    return img


def create_env_depth(height, width, frame_id, fov_factor, motion_intensity, perspective_offset):
    """Crée une depth map avec variations de caméra"""
    y, x = np.ogrid[:height, :width]
    
    # Point focal avec perspective
    center_x = width // 2
    center_y = int(height // 2 + perspective_offset * height)
    
    # Distance euclidienne avec facteur FOV
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2) / fov_factor
    
    # Profondeur de base (1-5m)
    depth_base = 1.0 + 4.0 * dist / np.max(dist)
    
    # Modulation temporelle
    depth_modulation = 0.3 * motion_intensity * np.sin(frame_id * 0.02)
    depth = depth_base + depth_modulation
    
    # Bruit selon qualité caméra
    noise = 0.05 * np.random.randn(height, width)
    depth = depth + noise
    depth = np.clip(depth, 0.1, 10.0)
    
    # Format TartanAir (mm * 1000)
    depth_int = (depth * 1000).astype(np.uint16)
    return depth_int


def create_env_seg(height, width, frame_id, object_count, env_name):
    """Crée une segmentation avec plus de classes selon l'environnement"""
    seg = np.zeros((height, width), dtype=np.uint8)
    
    # Fond selon environnement
    if "Night" in env_name:
        seg.fill(0)  # Nuit = fond noir (classe 0)
    else:
        seg.fill(1)  # Jour = fond clair (classe 1)
    
    # Objets avec classes différentes
    for obj_i in range(object_count):
        base_x = width * (0.2 + 0.6 * obj_i / max(1, object_count - 1))
        base_y = height * 0.5
        
        center_x = int(base_x + 30 * np.sin(frame_id * 0.05 + obj_i))
        center_y = int(base_y + 20 * np.cos(frame_id * 0.07 + obj_i))
        
        # Classe selon type d'objet
        if obj_i == 0:
            class_id = 2  # Objet mobile principal
            radius = 25
        elif obj_i == 1:
            class_id = 3  # Objet mobile secondaire
            radius = 20
        else:
            class_id = 4 + (obj_i % 2)  # Classes 4-5 pour autres objets
            radius = 15
        
        cv2.circle(seg, (center_x, center_y), radius, class_id, -1)
    
    return seg


def main():
    parser = argparse.ArgumentParser(description="Téléchargement élargi TartanAir synthétique")
    parser.add_argument("--root", default="data/tartanair-v2", help="Dossier racine")
    parser.add_argument("--envs", nargs="+", 
                       default=["ArchVizTinyHouseDay", "ArchVizTinyHouseNight"],
                       help="Environnements à générer")
    parser.add_argument("--difficulties", nargs="+", 
                       default=["easy", "hard"],
                       help="Niveaux de difficulté")
    parser.add_argument("--cameras", nargs="+",
                       default=["lcam_bottom", "lcam_front"],
                       help="Caméras à générer")
    parser.add_argument("--frames_per_env", type=int, default=100,
                       help="Nombre de frames par combinaison env/diff/cam")
    
    args = parser.parse_args()
    
    total = create_expanded_synthetic_data(
        args.root, args.envs, args.difficulties, 
        args.cameras, args.frames_per_env
    )
    
    print(f"🎉 Données synthétiques élargies créées: {total} frames au total")


if __name__ == "__main__":
    main()