# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 16:45:06 2025

modelo:  BiGru+ ensemble de 5
"""
import os
import numpy as np
import pandas as pd
import wfdb
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, roc_auc_score

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
#---------------------
#DIVIDIR TRAIN/TEST/VAL
#-----------------------
# 1-8 → train, 9 → val, 10 → test
X_train = X[df.strat_fold < 9]
y_train = Y[df.strat_fold < 9]
X_val = X[df.strat_fold == 9]
y_val = Y[df.strat_fold == 9]
X_test = X[df.strat_fold == 10]
y_test = Y[df.strat_fold == 10]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ------------------------
# DATA AUGMENTATION ESPECÍFICO (ENTRENAMIENTO)
# ------------------------

def ecg_augmentation_flexible(signal, label, mlb, hyp_class='HYP'):
    """
    Aplica augmentación suave a todas las muestras,
    y augmentación más fuerte si la clase HYP está presente.

    Parámetros:
        signal: tensor (12, T)
        label: tensor (num_classes,)
        mlb: fitted MultiLabelBinarizer con clases
        hyp_class: nombre de la clase HYP (por defecto 'HYP')

    Retorna:
        signal augmentado (tensor)
    """
    is_hyp = label[mlb.classes_.tolist().index(hyp_class)] == 1

    # Probabilidad base de augmentación
    base_prob = 0.2         # todas las clases
    hyp_extra_prob = 0.6    # si es HYP

    prob = base_prob + (hyp_extra_prob if is_hyp else 0)

    if np.random.random() > prob:
        return signal

    # -------- Augmentaciones comunes --------
    # 1. Time shifting
    if np.random.random() < 0.5:
        shift = np.random.randint(-20, 20)
        signal = torch.roll(signal, shifts=shift, dims=-1)

    # 2. Amplitude scaling
    if np.random.random() < 0.3:
        scale = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
        signal = signal * scale

    # 3. Gaussian noise
    if np.random.random() < 0.3:
        noise = torch.randn_like(signal) * 0.005
        signal = signal + noise

    # -------- Augmentaciones más intensas solo para HYP --------
    if is_hyp:
        # Stretching temporal leve
        if np.random.random() < 0.5:
            stretch_factor = torch.FloatTensor(1).uniform_(0.95, 1.05).item()
            original_length = signal.shape[-1]
            new_length = int(original_length * stretch_factor)
            if new_length != original_length:
                signal = torch.nn.functional.interpolate(
                    signal.unsqueeze(0), size=original_length, mode='linear'
                ).squeeze(0)

        # Reaplicar shift más fuerte
        if np.random.random() < 0.5:
            shift = np.random.randint(-30, 30)
            signal = torch.roll(signal, shifts=shift, dims=-1)

    return signal



#------------------
#DATA LOADER
#------------------
from torch.utils.data import Dataset, DataLoader
class ECGDataset(Dataset):
    def __init__(self, ecg_array, labels_array, training=False):
        self.ecg_data = torch.tensor(ecg_array, dtype=torch.float32)
        self.labels = torch.tensor(labels_array, dtype=torch.float32)
        self.training = training  # ✅ Añadir este atributo

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        ecg = self.ecg_data[idx]          # (12, T)
        label = self.labels[idx]          # (num_classes,)

        ecg = normalize_per_lead(ecg.unsqueeze(0)).squeeze(0)

        # Solo aplicar data augmentation si estamos entrenando
        if self.training:
            ecg = ecg_augmentation_flexible(ecg, label, mlb)

        return ecg, label

#NORMALIZACION
def normalize_per_lead(ecg_data):
         """
         Normaliza por lead (canal), para un batch o una única muestra.
         ecg_data: tensor (batch_size, 12, T) o (1, 12, T)
         """
         normalized = torch.zeros_like(ecg_data)
         for lead in range(ecg_data.shape[1]):
             lead_data = ecg_data[:, lead, :]
             mean = lead_data.mean(dim=1, keepdim=True)
             std = lead_data.std(dim=1, keepdim=True) + 1e-8
             normalized[:, lead, :] = (lead_data - mean) / std
         return normalized
train_loader = DataLoader(ECGDataset(X_train, y_train, training=True), batch_size=64, shuffle=True)
val_loader = DataLoader(ECGDataset(X_val, y_val, training=False), batch_size=64)
test_loader = DataLoader(ECGDataset(X_test, y_test, training=False), batch_size=64)

# ------------------
# MODELO Bi-GRU
# ------------------
class BiGRU_ECG(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, dropout=0.5, num_classes=5):  # fijo num_classes a 5
        super(BiGRU_ECG, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.bigru = nn.GRU(
            input_size=input_size,          # ✅ ahora usa 12 derivaciones
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.sigmoid = nn.Sigmoid()  # Multi-label classification

    def forward(self, x):
        # x: (batch_size, seq_len, input_size) → (B, T, 12)
        gru_out, _ = self.bigru(x)
        out_forward = gru_out[:, -1, :self.hidden_size]
        out_backward = gru_out[:, 0, self.hidden_size:]
        out = torch.cat((out_forward, out_backward), dim=1)
        out = self.fc(out)
        return self.sigmoid(out)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=10):
    best_model_wts = None
    best_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.permute(0, 2, 1).to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.permute(0, 2, 1).to(device)
                labels = labels.to(device).float()
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            torch.save(model.state_dict(), os.path.join(outputfolder, "best_bigru.pt"))
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"⏹️ Early stopping activado en la época {epoch+1}")
            break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    else:
        print("⚠️ No se encontró un modelo mejor durante el entrenamiento.")

    return model

# ------------------
# DEFINIR MODELO CON 12 DERIVACIONES
# ------------------
model = BiGRU_ECG(
    input_size=12,
    hidden_size=128,
    num_layers=2,
    dropout=0.5,
    num_classes=len(mlb.classes_)  # ← 5 superclases: CD, HYP, MI, NORM, STTC
).to(device)

#FUNCION DE EVALUACION
def evaluate_bigru_with_tta(model, dataloader, device, window_size=250, overlap=0.5, threshold=0.5):
    model.eval()
    all_probs = []
    all_targets = []
    step = int(window_size * (1 - overlap))

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluando Bi-GRU con TTA"):
            labels = labels.cpu().numpy()
            batch_preds = []

            for ecg in inputs:
                windows = []
                for start in range(0, ecg.shape[1] - window_size + 1, step):
                    segment = ecg[:, start:start + window_size]
                    if segment.shape[1] == window_size:
                        windows.append(segment.T.unsqueeze(0))  # (1, T, 12)

                if windows:
                    windows = torch.cat(windows, dim=0).to(device)  # (num_windows, T, 12)
                    preds = model(windows)
                    max_pred = preds.max(dim=0).values  # (num_classes,)
                else:
                    ecg_input = ecg.T.unsqueeze(0).to(device)  # (1, T, 12)
                    max_pred = model(ecg_input).squeeze(0)

                batch_preds.append(max_pred.cpu().numpy())

            all_probs.append(np.stack(batch_preds))
            all_targets.append(labels)

    y_probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    y_pred = (y_probs > threshold).astype(int)
    print("\n📊 Reporte de clasificación con TTA:")
    print(classification_report(y_true, y_pred, target_names=mlb.classes_))
    try:
        auc_macro = roc_auc_score(y_true, y_probs, average='macro')
        print(f"AUC macro con TTA: {auc_macro:.4f}")
    except ValueError:
        print("No se pudo calcular el AUC (probablemente falta alguna clase).")

    return y_probs, y_true

#ENTRENAMIENTO Y EVALUACION FINAL
# Pérdida ponderada similar a la que usas con InceptionTime (ajusta si quieres)
class_weights = torch.tensor([
    1.0,    # CD
    8.0,    # HYP (penalización fuerte)
    1.5,    # MI
    0.5,    # NORM
    2.5     # STTC
]).to(device)

criterion = nn.BCELoss()  # Usamos salida con sigmoid, así que usamos BCELoss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#  Entrenar el modelo
model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50)

#  Evaluar en test set
y_probs_bigru, y_true_bigru = evaluate_bigru_with_tta(model, test_loader, device)

# Resultados con threshold 0.5
y_pred_bigru = (y_probs_bigru > 0.5).astype(int)

from sklearn.metrics import classification_report
print("\nResultados Bi-GRU con threshold 0.5:")
print(classification_report(y_true_bigru, y_pred_bigru, target_names=mlb.classes_))

#GUARDAR RESULTADOS
np.save(os.path.join(outputfolder, 'y_probs_bigru.npy'), y_probs_bigru)
np.save(os.path.join(outputfolder, 'y_true_bigru.npy'), y_true_bigru)

#-----------------
#ENSEMBLE
#-----------------

def train_bigru_ensemble(num_models=5, base_seed=100):
    models = []
    individual_results = []
    
    for i in range(num_models):
        print(f"\n🔁 Entrenando Bi-GRU modelo {i+1}/{num_models}...")
        seed = base_seed + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = BiGRU_ECG(
            input_size=12,
            hidden_size=128,
            num_layers=2,
            dropout=0.5,
            num_classes=len(mlb.classes_)
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50)

        y_probs, y_true = evaluate_bigru_with_tta(model, test_loader, device)
        auc = roc_auc_score(y_true, y_probs, average="macro")

        individual_results.append({
            'model': model,
            'seed': seed,
            'predictions': y_probs,
            'auc_macro': auc
        })

        print(f"✅ AUC macro modelo {i+1}: {auc:.4f}")
        models.append(model)

    return models, individual_results, y_true

#EVALUAR EL ENSEMBLE
def evaluate_bigru_ensemble(individual_results, y_true, mlb, save_name='bigru_ensemble'):
    print("\n🔎 Evaluando Ensemble de Bi-GRU...")

    all_preds = [result['predictions'] for result in individual_results]
    y_probs_ensemble = np.mean(all_preds, axis=0)

    y_pred = (y_probs_ensemble > 0.5).astype(int)

    print("\n📊 Classification Report (threshold 0.5):")
    print(classification_report(y_true, y_pred, target_names=mlb.classes_))

    auc_macro = roc_auc_score(y_true, y_probs_ensemble, average='macro')
    auc_per_class = roc_auc_score(y_true, y_probs_ensemble, average=None)
    print(f"\n📈 AUC macro (ensemble): {auc_macro:.4f}")
    for i, cls in enumerate(mlb.classes_):
      print(f"AUC {cls}: {auc_per_class[i]:.4f}")55

    # Guardar
    np.save(os.path.join(outputfolder, f'y_probs_B{save_name}.npy'), y_probs_ensemble)
    np.save(os.path.join(outputfolder, f'y_true_B{save_name}.npy'), y_true)

    return y_probs_ensemble
if __name__ == "__main__":
    models_bigru, individual_results_bigru, y_true_bigru = train_bigru_ensemble(num_models=5)
    y_probs_ensemble_bigru = evaluate_bigru_ensemble(individual_results_bigru, y_true_bigru, mlb)

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
# Optimización de thresholds
optimal_thresholds = optimize_thresholds(y_true_bigru, y_probs_ensemble_bigru, mlb.classes_)

print("\nThresholds óptimos por clase:")
for cls, thr in optimal_thresholds.items():
    print(f"{cls}: {thr:.2f}")

# Aplicar thresholds óptimos
y_pred_opt = np.zeros_like(y_probs_ensemble_bigru)
for i, cls in enumerate(mlb.classes_):
    y_pred_opt[:, i] = (y_probs_ensemble_bigru[:, i] > optimal_thresholds[cls]).astype(int)

print("\n📊 Reporte con thresholds optimizados:")
print(classification_report(y_true_bigru, y_pred_opt, target_names=mlb.classes_))
# ============================================================
#  CURVA PRECISIÓN-RECALL  (AUC-PR)  PARA MULTI-LABEL
# ============================================================
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def plot_pr_curves(y_true, y_probs, class_names, save_path=None):
    """
    Dibuja curvas P-R para cada clase y macro/micro promedio.
    """
    n_classes = y_true.shape[1]
    ap_per_class = {}
    # Micro-average
    precision_micro, recall_micro, _ = precision_recall_curve(y_true.ravel(), y_probs.ravel())
    ap_micro = average_precision_score(y_true, y_probs, average="micro")

    # Macro-average
    aps = []
    plt.figure(figsize=(8, 6))
    plt.plot(recall_micro, precision_micro, label=f"Micro-average (AP = {ap_micro:.3f})", linewidth=2, color="black")

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        ap = average_precision_score(y_true[:, i], y_probs[:, i])
        ap_per_class[class_names[i]] = ap
        aps.append(ap)
        plt.plot(recall, precision, label=f"{class_names[i]} (AP = {ap:.3f})")

    ap_macro = np.mean(aps)
    ap_per_class["macro"] = ap_macro
    ap_per_class["micro"] = ap_micro


    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas Precisión-Recall (Ensemble Bi-GRU)")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📁 Gráfico guardado en {save_path}")
    else:
        plt.show()

    return ap_per_class
# Mostrar en pantalla
plot_pr_curves(y_true_bigru, y_probs_ensemble_bigru, mlb.classes_)

