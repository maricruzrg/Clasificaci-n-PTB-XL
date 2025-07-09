# -*- coding: utf-8 -*-
"""
Visualización casos correctamente e incorrectamente clasificados por el modelo ResNet1D
"""
import os
import numpy as np
import pandas as pd
import wfdb
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from tqdm import tqdm
# ------------------------
# 1. CONFIGURACIÓN
# ------------------------
datafolder = r'C:\Users\maric\Desktop\ptbxl\ptbxl'
outputfolder = r'C:\Users\maric\Desktop\TFM\output'
os.makedirs(outputfolder, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------
# 2. CARGA DE DATOS
# ------------------------
df = pd.read_csv(os.path.join(datafolder, 'ptbxl_database.csv'), index_col='ecg_id')
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
scp_statements = pd.read_csv(os.path.join(datafolder, 'scp_statements.csv'), index_col=0)
diagnostic_codes = scp_statements[scp_statements.diagnostic == 1].index

# Filtrar sólo los códigos diagnósticos
df['diagnostic_codes'] = df['scp_codes'].apply(lambda x: [code for code in x if code in diagnostic_codes])
df = df[df['diagnostic_codes'].map(len) > 0]

# Mapear códigos a superclases diagnósticas
superclass_map = scp_statements['diagnostic_class'].dropna().to_dict()
df['superclasses'] = df['diagnostic_codes'].apply(
    lambda codes: list(set(superclass_map[c] for c in codes if c in superclass_map))
)

# Filtrar por ECGs con al menos una superclase válida
df = df[df['superclasses'].map(len) > 0]

# Binarizar etiquetas (superclases)
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['superclasses'])

print("Superclases:", mlb.classes_)

def load_signal(row, sampling_rate=100):
    path = os.path.join(datafolder, row['filename_lr' if sampling_rate == 100 else 'filename_hr'])
    signal, _ = wfdb.rdsamp(path)
    return signal.T

X = np.array([load_signal(row) for _, row in df.iterrows()])
# Reconstruir X_test para que coincida con y_probs / y_true
X_test = X[df.strat_fold == 10]

import matplotlib.pyplot as plt

# === Carga de archivos ===
threshold = 0.5  # Umbral de clasificación
class_names = list(mlb.classes_)  # Tomar las superclases del binarizador

y_probs = np.load(os.path.join(outputfolder, 'y_probs_ensembleECG.npy'))
y_true = np.load(os.path.join(outputfolder, 'ytrue_ensembleECG.npy'))

# Binarización
y_pred = (y_probs >= threshold).astype(int)

# Crear carpeta de salida
os.makedirs("casos_clasificados", exist_ok=True)

# === Iterar por clase ===
# === Iterar por clase ===
for class_idx, class_name in enumerate(class_names):
    true_class = y_true[:, class_idx]
    pred_class = y_pred[:, class_idx]

    correct_indices = np.where((true_class == 1) & (pred_class == 1))[0]
    incorrect_indices = np.where((true_class == 1) & (pred_class == 0))[0]

    # Ejemplo correcto
    if len(correct_indices) > 0:
        idx = correct_indices[0]
        plt.figure(figsize=(10, 3))
        plt.plot(X_test[idx][1])  # Lead II
        plt.title(f"{class_name} - Clasificación correcta (Derivación II)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud (mV)")
        plt.tight_layout()
        plt.show()

    # Ejemplo incorrecto
    if len(incorrect_indices) > 0:
        idx = incorrect_indices[0]
        plt.figure(figsize=(10, 3))
        plt.plot(X_test[idx][1])  # Lead II
        plt.title(f"{class_name} - Clasificación incorrecta (Derivación II)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud (mV)")
        plt.tight_layout()
        plt.show()

