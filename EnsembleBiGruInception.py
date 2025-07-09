
"""
Late Fusion (Promedio) entre InceptionTime y Bi-GRU
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import json

# ----------------------
# Configuración
# ----------------------
outputfolder = r'C:\Users\maric\Desktop\TFM\output'

# ----------------------
# Cargar probabilidades y ground truth
# ----------------------
y_probs_bigru = np.load(os.path.join(outputfolder, 'y_probs_bigru_ensemble.npy'))
y_true_bigru = np.load(os.path.join(outputfolder, 'y_true_bigru_ensemble.npy'))

y_probs_inception = np.load(os.path.join(outputfolder, 'y_probs_ensemble.npy'))  # InceptionTime
y_true_inception = np.load(os.path.join(outputfolder, 'ytrue_ensemble.npy'))

# Verificar consistencia
assert np.allclose(y_true_bigru, y_true_inception), " Los y_true no coinciden entre modelos"
y_true = y_true_bigru  # usar uno solo

# ----------------------
# Cargar nombres de clases y thresholds
# ----------------------
mlb_classes = ['CD', 'HYP', 'MI', 'NORM', 'STTC'] 

# ----------------------
# Fusión por promedio
# ----------------------
y_probs_fused = (y_probs_bigru + y_probs_inception) / 2

# ----------------------
# Optimizar thresholds sobre el promedio
# ----------------------
from sklearn.metrics import f1_score

def optimize_thresholds(y_true, y_pred_proba, class_names):
    thresholds = {}
    for i, class_name in enumerate(class_names):
        best_threshold = 0.5
        best_f1 = 0
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred_class = (y_pred_proba[:, i] > threshold).astype(int)
            f1 = f1_score(y_true[:, i], y_pred_class)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds[class_name] = best_threshold
    return thresholds

optimal_thresholds = optimize_thresholds(y_true, y_probs_fused, mlb_classes)

# ----------------------
# Aplicar thresholds óptimos
# ----------------------
y_pred_fused = np.zeros_like(y_probs_fused)
for i, class_name in enumerate(mlb_classes):
    thr = optimal_thresholds[class_name]
    y_pred_fused[:, i] = (y_probs_fused[:, i] > thr).astype(int)

# ----------------------
# Evaluación
# ----------------------
print("\n Clasificación del Ensemble (Promedio + Thresholds Óptimos):")
print(classification_report(y_true, y_pred_fused, target_names=mlb_classes))

auc_macro = roc_auc_score(y_true, y_probs_fused, average="macro")
print(f"AUC macro (ensemble): {auc_macro:.4f}")

# ----------------------
# Guardar resultados
# ----------------------
np.save(os.path.join(outputfolder, 'y_probs_fused.npy'), y_probs_fused)
np.save(os.path.join(outputfolder, 'y_pred_fused.npy'), y_pred_fused)
