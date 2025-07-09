# -*- coding: utf-8 -*-
"""
RESNETWANG ENSEMBLE 5 MODELOS
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
        # Interpolamos a la nueva longitud (stretch), luego reescalamos de vuelta a original_length
         signal = signal.unsqueeze(0)  # (1, 12, T)
         signal = F.interpolate(signal, size=new_length, mode='linear', align_corners=False)
         signal = F.interpolate(signal, size=original_length, mode='linear', align_corners=False)
         signal = signal.squeeze(0)


        # Reaplicar shift más fuerte
        if np.random.random() < 0.5:
            shift = np.random.randint(-30, 30)
            signal = torch.roll(signal, shifts=shift, dims=-1)

    return signal

# ------------------------
# DATALOADER
# ------------------------

import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, ecg_array, labels_array, training=False):
        """
        ecg_array: np.array o torch.tensor de forma (N, 12, T)
        labels_array: np.array o torch.tensor de forma (N, num_classes)
        training: si True, aplica augmentations
        """
        self.ecg_data = torch.tensor(ecg_array, dtype=torch.float32) if not torch.is_tensor(ecg_array) else ecg_array
        self.labels = torch.tensor(labels_array, dtype=torch.float32) if not torch.is_tensor(labels_array) else labels_array
        self.training = training

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
            ecg = self.ecg_data[idx]          # (12, T)
            label = self.labels[idx]          # (num_classes,)
        
            ecg = normalize_per_lead(ecg.unsqueeze(0)).squeeze(0)

        # Solo aplicar data augmentation si:
        # 1. Es fase de entrenamiento
        # 2. La clase HYP (índice 1 en mlb.classes_) está presente
            if self.training:
              ecg = ecg_augmentation_flexible(ecg,label,mlb)

            return ecg, label

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

from torch.utils.data import Dataset, DataLoader

train_dataset = ECGDataset(X_train, y_train, training=True)
val_dataset = ECGDataset(X_val, y_val, training=False)
test_dataset = ECGDataset(X_test, y_test, training=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

#7. Crear modelo

import torch.nn as nn
import torch.nn.functional as F

# Define ResNet1D Wang con bloques básicos
def conv(in_planes, out_planes, stride=1, kernel_size=3):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size - 1) // 2, bias=False)

class BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, kernel_size=[5, 3], downsample=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size // 2 + 1]
        self.conv1 = conv(inplanes, planes, stride=stride, kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes, kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def create_head1d(in_ftrs, nc, **kwargs):
    return nn.Sequential(
        nn.AdaptiveAvgPool1d(1),
        Flatten(),
        nn.Linear(in_ftrs, nc)
    )

class ResNet1d(nn.Sequential):
    def __init__(self, block, layers, kernel_size=3, num_classes=5, input_channels=12, inplanes=128, fix_feature_dim=True, kernel_size_stem=7, stride_stem=1, pooling_stem=False, stride=2):
        self.inplanes = inplanes
        layers_tmp = []
        layers_tmp += [
            nn.Conv1d(input_channels, inplanes, kernel_size=kernel_size_stem, stride=stride_stem,
                      padding=(kernel_size_stem - 1) // 2, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True)
        ]
        if pooling_stem:
            layers_tmp.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        for i, l in enumerate(layers):
            planes = inplanes if fix_feature_dim else (2**i) * inplanes
            s = stride if i > 0 else 1
            layers_tmp.append(self._make_layer(block, planes, l, stride=s, kernel_size=kernel_size))

        n_feats = (inplanes if fix_feature_dim else (2**len(layers)) * inplanes) * block.expansion
        head = create_head1d(n_feats, nc=num_classes)
        layers_tmp.append(head)
        super().__init__(*layers_tmp)

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=[5, 3]):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion)
            )
        layers = [block(self.inplanes, planes, stride, kernel_size, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

def resnet1d_wang(**kwargs):
    kwargs.setdefault("kernel_size", [5, 3])
    kwargs.setdefault("kernel_size_stem", 7)
    kwargs.setdefault("stride_stem", 1)
    kwargs.setdefault("pooling_stem", False)
    kwargs.setdefault("inplanes", 128)
    return ResNet1d(BasicBlock1d, [1, 1, 1], **kwargs)


#8. Entrenamiento

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet1d_wang(input_channels=12, num_classes=5).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

class_weights = torch.tensor([
    1.0,    # CD
    5.0,    # HYP (más agresivo)
    1.5,    # MI
    0.6,    # NORM (reducir más)
    2.5     # STTC (incrementar)
])
#------------------
#EVALUACION
#-----------------
from sklearn.metrics import classification_report, roc_auc_score, f1_score

def evaluate_model_with_tta(model, dataloader, mlb, device, threshold=0.5):
    """
    Evalúa el modelo con Test-Time Augmentation (TTA)
    """
    model.eval()
    all_predictions = []
    all_targets = []

    window_size = 250  # 2.5s a 100Hz
    overlap = 0.5
    step = int(window_size * (1 - overlap))

    with torch.no_grad():
        for batch_data, batch_targets in tqdm(dataloader, desc="Evaluación con TTA"):
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.cpu().numpy()
            batch_preds = []

            for sample in batch_data:
                sample_preds = []
                for start in range(0, sample.shape[-1] - window_size + 1, step):
                    window = sample[:, start:start + window_size]
                    if window.shape[-1] == window_size:
                        pred = torch.sigmoid(model(window.unsqueeze(0))).cpu()
                        sample_preds.append(pred)

                if sample_preds:
                    stacked = torch.stack(sample_preds, dim=0)
                    final_pred = torch.max(stacked, dim=0).values.squeeze(0)
                else:
                    final_pred = torch.sigmoid(model(sample.unsqueeze(0))).squeeze(0).cpu()

                batch_preds.append(final_pred)

            # ✅ Convertir lista de predicciones a array numpy
            batch_preds_np = torch.stack(batch_preds).numpy()

            all_predictions.extend([p.numpy() for p in batch_preds])
            all_targets.extend(batch_targets)

    y_probs = np.stack(all_predictions, axis=0)
    y_true = np.stack(all_targets, axis=0)

    y_pred = (y_probs > threshold).astype(int)

    # -------------------
    # MÉTRICAS DE SALIDA
    # -------------------
    print("\n--- MÉTRICAS DE CLASIFICACIÓN (con TTA) ---")
    print(classification_report(y_true, y_pred, target_names=mlb.classes_))

    try:
        auc_macro = roc_auc_score(y_true, y_probs, average="macro")
        auc_per_class = roc_auc_score(y_true, y_probs, average=None)

        print(f"AUC macro promedio: {auc_macro:.4f}")
        for i, class_name in enumerate(mlb.classes_):
            print(f"AUC {class_name}: {auc_per_class[i]:.4f}")
    except ValueError as e:
        print(f"No se pudo calcular el AUC: {e}")

    return y_probs, y_true

class_weights = class_weights.to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

import time
import torch.optim as optim
history = []
best_auc = 0.0
save_path = os.path.join(outputfolder, "best_model.pt")

optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 100
patience = 10  # número de épocas sin mejora antes de detener
counter = 0    # cuenta de épocas sin mejora
early_stop = False


def train_model(model, train_loader, test_loader, mlb, criterion, optimizer, device, save_path, num_epochs=100, patience=10):
    best_auc = 0.0
    counter = 0
    history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Entrenando"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        y_probs, y_true = evaluate_model_with_tta(model, test_loader, mlb, device)
        try:
            auc_macro = roc_auc_score(y_true, y_probs, average="macro")
        except ValueError:
            auc_macro = 0.0

        history.append({'epoch': epoch+1, 'loss': epoch_loss, 'auc_macro': auc_macro})

        if auc_macro > best_auc:
            best_auc = auc_macro
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            if counter >= patience:
               print(f" Early stopping activado en la época {epoch+1} (mejor AUC: {best_auc:.4f})")
               break
         
    model.load_state_dict(torch.load(save_path))
    return model

#Implementar train emsemble
def ensemble_predict(individual_results):
    print("\n=== Calculando predicciones ensemble ===")
    all_predictions = [result['predictions'] for result in individual_results]
    ensemble_predictions = np.mean(all_predictions, axis=0)
    return ensemble_predictions
def optimize_thresholds(y_true, y_pred_proba):
    thresholds = {}
    class_names = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
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
def run_ensemble_experiment(num_models=5):
    print("Iniciando experimento ensemble...")
    models, individual_results, y_true = train_ensemble_models(num_models=num_models)
    ensemble_pred = ensemble_predict(individual_results)
    final_auc, thresholds = analyze_ensemble_results(individual_results, ensemble_pred, y_true)
    print(f"\nRESULTADO FINAL: AUC macro ensemble = {final_auc:.4f}")
    return models, ensemble_pred, y_true, thresholds

def analyze_ensemble_results(individual_results, ensemble_pred, y_true):
    print("\n" + "="*50)
    print("RESULTADOS FINALES DEL ENSEMBLE")
    print("="*50)

    aucs = []
    for result in individual_results:
        print(f"  Seed {result['seed']}: AUC macro = {result['macro_auc']:.4f}")
        aucs.append(result['macro_auc'])

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nResumen individuales:")
    print(f"  Media ± std: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Rango: {min(aucs):.4f} - {max(aucs):.4f}")

    ensemble_auc = roc_auc_score(y_true, ensemble_pred, average='macro')
    ensemble_auc_per_class = roc_auc_score(y_true, ensemble_pred, average=None)
    print(f"\nEnsemble:")
    print(f"  AUC macro: {ensemble_auc:.4f}")
    print(f"  AUC por clase [CD, HYP, MI, NORM, STTC]: {ensemble_auc_per_class}")
    improvement = ensemble_auc - max(aucs)
    print(f"  Mejora vs mejor individual: +{improvement:.4f}")

    optimal_thresholds = optimize_thresholds(y_true, ensemble_pred)
    print(f"\nThresholds óptimos: {optimal_thresholds}")

    y_pred_optimized = np.zeros_like(ensemble_pred)
    class_names = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
    for i, class_name in enumerate(class_names):
        threshold = optimal_thresholds[class_name]
        y_pred_optimized[:, i] = (ensemble_pred[:, i] > threshold).astype(int)

    print(f"\nClassification Report (Ensemble con thresholds optimizados):")
    print(classification_report(y_true, y_pred_optimized,
                                target_names=class_names, digits=3))
    return ensemble_auc, optimal_thresholds

def train_ensemble_models(num_models=3, base_seed=42):
    models = []
    individual_results = []
    
    for i in range(num_models):
        seed = base_seed + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = resnet1d_wang(input_channels=12, num_classes=5).to(device)



        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        save_path = os.path.join(outputfolder, f"model_seed_{seed}.pt")
        trained_model = train_model(
            model, train_loader, test_loader, mlb,
            criterion, optimizer, device, save_path
        )

        y_pred_proba, y_true = evaluate_model_with_tta(trained_model, test_loader, mlb, device)

        auc_macro = roc_auc_score(y_true, y_pred_proba, average='macro')
        auc_per_class = roc_auc_score(y_true, y_pred_proba, average=None)

        result = {
            'seed': seed,
            'model_idx': i,
            'macro_auc': auc_macro,
            'auc_per_class': auc_per_class,
            'predictions': y_pred_proba
        }
        individual_results.append(result)
        models.append(trained_model)
        
        print(f"[Modelo {i+1}] AUC macro: {auc_macro:.4f}")
    
    return models, individual_results, y_true

# Dentro de __main__
if __name__ == "__main__":
    models, ensemble_predictions, y_true, optimal_thresholds = run_ensemble_experiment(num_models=5)

    # Guardado seguro con ruta absoluta
    np.save(os.path.join(outputfolder, 'y_probs_ensembleECG.npy'), ensemble_predictions)
    np.save(os.path.join(outputfolder, 'ytrue_ensembleECG.npy'), y_true)

print("Shape ensemble_predictions:", ensemble_predictions.shape)  # debería ser (2163, 5)
print("Shape y_true:", y_true.shape)                              # también (2163, 5)

# ============================================================
#  CURVA PRECISIÓN-RECALL  (AUC-PR)  PARA MULTI-LABEL
# ============================================================
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curves(y_true, y_probs, class_names, save_path=None):
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
    ap_per_class["micro"] = ap_micro
    ap_per_class["macro"] = ap_macro

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas Precisión-Recall")
    plt.legend(loc="lower left")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

    return ap_per_class

# -------------------------------
# ⚠️ Asegúrate de que estas variables existen:
# y_true, ensemble_predictions, mlb
# -------------------------------
class_names = list(mlb.classes_)

print("\nGenerando curvas P-R y métricas AUC-PR...")
ap_dict = plot_pr_curves(
    y_true=y_true,
    y_probs=ensemble_predictions,
    class_names=class_names,
    save_path=os.path.join(outputfolder, "pr_curves.png")  # opcional
)

print("\n=====  AUC-PR (Average Precision)  =====")
for cls, ap in ap_dict.items():
    print(f"{cls:>6}: {ap:.4f}")


