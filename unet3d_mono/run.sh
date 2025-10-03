#!/usr/bin/env bash
# Script de lancement pour éviter ModuleNotFoundError
# Force le PYTHONPATH à la racine du projet

PYTHONPATH=. python train.py configs/indoor_lowcam.yaml