# -*- coding: utf-8 -*-
"""
X-ResNet1D-101+ ensemble 5 modelos
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

# Preprocesamiento
import h5py

def preprocess_and_save(X, filepath, target_length=5000):
    with h5py.File(filepath, 'w') as f:
        dset = f.create_dataset("X", shape=(len(X), 12, target_length), dtype=np.float32)
        for i, s in enumerate(tqdm(X, desc="Preprocesando")):
            s = (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-7)
            if s.shape[1] < target_length:
                s = np.pad(s, ((0, 0), (0, target_length - s.shape[1])), 'constant')
            else:
                s = s[:, :target_length]
            dset[i] = s.astype(np.float32)

h5_filepath = os.path.join(outputfolder, 'X_preprocessed.h5')
preprocess_and_save(X, h5_filepath)
h5_file = h5py.File(h5_filepath, 'r')
X_all = h5_file['X']

import torch
from torch.utils.data import Dataset, DataLoader

class H5Dataset(Dataset):
    def __init__(self, h5_data, labels, indices):
        self.h5_data = h5_data
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.h5_data[i]
        y = self.labels[i]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Abrir archivo sin cerrarlo
h5_file = h5py.File(h5_filepath, 'r')
X_all = h5_file['X']  # No usamos [:] para no cargar todo

# Dividir índices
train_idx = np.where(df['strat_fold'] <= 8)[0]
val_idx   = np.where(df['strat_fold'] == 9)[0]
test_idx  = np.where(df['strat_fold'] == 10)[0]
print(f"Shape de Y: {Y.shape}")
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
print(f"Total indices: {len(train_idx) + len(val_idx) + len(test_idx)}")

#Crear datset y dataloaders
batch_size = 64

train_dataset = H5Dataset(X_all, Y, train_idx)
val_dataset   = H5Dataset(X_all, Y, val_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)

#XRESNET MODEL

import torch

import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def create_head1d(in_features, nc, lin_ftrs=None, ps=0.5, bn_final=False, bn=True, act="relu", concat_pooling=True):
    """
    Crea una cabeza (head) para una red 1D para tareas de clasificación multietiqueta.
    """
    if lin_ftrs is None: lin_ftrs = [512]
    if isinstance(ps, float): ps = [ps/2] * len(lin_ftrs) + [ps]
    layers = []

    if concat_pooling:
        pool = nn.AdaptiveAvgPool1d(1)
        layers += [pool, Flatten()]
        in_features *= 1
    else:
        layers += [Flatten()]

    ftrs = [in_features] + lin_ftrs + [nc]
    for i in range(len(ftrs)-2):
        layers += [nn.Linear(ftrs[i], ftrs[i+1])]
        if bn: layers += [nn.BatchNorm1d(ftrs[i+1])]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(ps[i])]

    layers += [nn.Linear(ftrs[-2], ftrs[-1])]
    if bn_final: layers += [nn.BatchNorm1d(ftrs[-1])]
    if act == "sigmoid":
        layers += [nn.Sigmoid()]
    elif act == "softmax":
        layers += [nn.Softmax(dim=1)]

    return nn.Sequential(*layers)


from enum import Enum
import re
#delegates
import inspect

def delegates(to=None, keep=False):
    "Decorator: replace `**kwargs` in signature with params from `to`"
    def _f(f):
        if to is None: to_f,from_f = f.__base__.__init__,f.__init__
        else:          to_f,from_f = to,f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {k:v for k,v in inspect.signature(to_f).parameters.items()
              if v.default != inspect.Parameter.empty and k not in sigd}
        sigd.update(s2)
        if keep: sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f

def store_attr(self, nms):
    "Store params named in comma-separated `nms` from calling context into attrs in `self`"
    mod = inspect.currentframe().f_back.f_locals
    for n in re.split(', *', nms): setattr(self,n,mod[n])

NormType = Enum('NormType', 'Batch BatchZero Weight Spectral Instance InstanceZero')

def _conv_func(ndim=2, transpose=False):
    "Return the proper conv `ndim` function, potentially `transposed`."
    assert 1 <= ndim <=3
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')

def init_default(m, func=nn.init.kaiming_normal_):
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func and hasattr(m, 'weight'): func(m.weight)
    with torch.no_grad():
        if getattr(m, 'bias', None) is not None: m.bias.fill_(0.)
    return m
    
def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    "Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0. if zero else 1.)
    return bn 

def BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm('BatchNorm', nf, ndim, zero=norm_type==NormType.BatchZero, **kwargs)

from torch.nn.utils import weight_norm, spectral_norm
from torch.nn import InstanceNorm1d, InstanceNorm2d, InstanceNorm3d

def InstanceNorm(nf, norm_type=NormType.Instance, ndim=1, **kwargs):
    if ndim == 1:
        return InstanceNorm1d(nf, **kwargs)
    elif ndim == 2:
        return InstanceNorm2d(nf, **kwargs)
    elif ndim == 3:
        return InstanceNorm3d(nf, **kwargs)
    else:
        raise NotImplementedError("Only 1D, 2D and 3D instance norm are supported.")


class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, ndim=2, norm_type=NormType.Batch, bn_1st=True,
                 act_cls=nn.ReLU, transpose=False, init=nn.init.kaiming_normal_, xtra=None, **kwargs):
        if padding is None: padding = ((ks-1)//2 if not transpose else 0)
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None: bias = not (bn or inn)
        conv_func = _conv_func(ndim, transpose=transpose)
        conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs), init)
        if   norm_type==NormType.Weight:   conv = weight_norm(conv)
        elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
        layers = [conv]
        act_bn = []
        if act_cls is not None: act_bn.append(act_cls())
        if bn: act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if inn: act_bn.append(InstanceNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st: act_bn.reverse()
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)

def AdaptiveAvgPool(sz=1, ndim=2):
    "nn.AdaptiveAvgPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"AdaptiveAvgPool{ndim}d")(sz)

def MaxPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False):
    "nn.MaxPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"MaxPool{ndim}d")(ks, stride=stride, padding=padding)

def AvgPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False):
    "nn.AvgPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"AvgPool{ndim}d")(ks, stride=stride, padding=padding, ceil_mode=ceil_mode)

# Módulo Squeeze-and-Excitation
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16, act_cls=nn.ReLU):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            act_cls(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
# Atención propia simplificada (1D)
class SimpleSelfAttention(nn.Module):
    def __init__(self, n_channels, ks=1, sym=False):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, n_channels, ks, padding=ks//2, groups=n_channels)
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.sym = sym

    def forward(self, x):
        # x: (B, C, L)
        attn = self.conv(x)
        if self.sym:
            attn = (attn + attn.transpose(1, 2)) / 2
        return x + self.gamma * attn

class ResBlock(nn.Module):
    "Resnet block from `ni` to `nh` with `stride`"
    @delegates(ConvLayer.__init__)
    def __init__(self, expansion, ni, nf, stride=1, kernel_size=3, groups=1, reduction=None, nh1=None, nh2=None, dw=False, g2=1,
                 sa=False, sym=False, norm_type=NormType.Batch, act_cls=nn.ReLU, ndim=2,
                 pool=AvgPool, pool_first=True, **kwargs):
        super().__init__()
        norm2 = (NormType.BatchZero if norm_type==NormType.Batch else
                 NormType.InstanceZero if norm_type==NormType.Instance else norm_type)
        if nh2 is None: nh2 = nf
        if nh1 is None: nh1 = nh2
        nf,ni = nf*expansion,ni*expansion
        k0 = dict(norm_type=norm_type, act_cls=act_cls, ndim=ndim, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, ndim=ndim, **kwargs)
        layers  = [ConvLayer(ni,  nh2, kernel_size, stride=stride, groups=ni if dw else groups, **k0),
                   ConvLayer(nh2,  nf, kernel_size, groups=g2, **k1)
        ] if expansion == 1 else [
                   ConvLayer(ni,  nh1, 1, **k0),
                   ConvLayer(nh1, nh2, kernel_size, stride=stride, groups=nh1 if dw else groups, **k0),
                   ConvLayer(nh2,  nf, 1, groups=g2, **k1)]
        self.convs = nn.Sequential(*layers)
        convpath = [self.convs]
        if reduction: convpath.append(SEModule(nf, reduction=reduction, act_cls=act_cls))
        if sa: convpath.append(SimpleSelfAttention(nf,ks=1,sym=sym))
        self.convpath = nn.Sequential(*convpath)
        idpath = []
        if ni!=nf: idpath.append(ConvLayer(ni, nf, 1, act_cls=None, ndim=ndim, **kwargs))
        if stride!=1: idpath.insert((1,0)[pool_first], pool(2, ndim=ndim, ceil_mode=True))
        self.idpath = nn.Sequential(*idpath)
        self.act = nn.ReLU(inplace=True) if act_cls is nn.ReLU else act_cls()

    def forward(self, x): return self.act(self.convpath(x) + self.idpath(x))

######################### adapted from vison.models.xresnet
def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d, nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

class XResNet1d(nn.Sequential):
    @delegates(ResBlock)
    def __init__(self, block, expansion, layers, p=0.0, input_channels=3, num_classes=1000, stem_szs=(32,32,64),kernel_size=5,kernel_size_stem=5,
                 widen=1.0, sa=False, act_cls=nn.ReLU, lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True, **kwargs):
        store_attr(self, 'block,expansion,act_cls')
        stem_szs = [input_channels, *stem_szs]
        stem = [ConvLayer(stem_szs[i], stem_szs[i+1], ks=kernel_size_stem, stride=2 if i==0 else 1, act_cls=act_cls, ndim=1)
                for i in range(3)]

        #block_szs = [int(o*widen) for o in [64,128,256,512] +[256]*(len(layers)-4)]
        block_szs = [int(o*widen) for o in [64,64,64,64] +[32]*(len(layers)-4)]
        block_szs = [64//expansion] + block_szs
        blocks = [self._make_layer(ni=block_szs[i], nf=block_szs[i+1], blocks=l,
                                   stride=1 if i==0 else 2, kernel_size=kernel_size, sa=sa and i==len(layers)-4, ndim=1, **kwargs)
                  for i,l in enumerate(layers)]

        head = create_head1d(block_szs[-1]*expansion, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        
        super().__init__(
            *stem, nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            *blocks,
            head,
        )
        init_cnn(self)

    def _make_layer(self, ni, nf, blocks, stride, kernel_size, sa, **kwargs):
        return nn.Sequential(
            *[self.block(self.expansion, ni if i==0 else nf, nf, stride=stride if i==0 else 1,
                      kernel_size=kernel_size, sa=sa and i==(blocks-1), act_cls=self.act_cls, **kwargs)
              for i in range(blocks)])
    
    def get_layer_groups(self):
        return (self[3],self[-1])
    
    def get_output_layer(self):
        return self[-1][-1]
        
    def set_output_layer(self,x):
        self[-1][-1]=x


#xresnets
def _xresnet1d(expansion, layers, **kwargs):
    return XResNet1d(ResBlock, expansion, layers, **kwargs)
    
def xresnet1d18 (**kwargs): return _xresnet1d(1, [2, 2,  2, 2], **kwargs)
def xresnet1d34 (**kwargs): return _xresnet1d(1, [3, 4,  6, 3], **kwargs)
def xresnet1d50 (**kwargs): return _xresnet1d(4, [3, 4,  6, 3], **kwargs)
def xresnet1d101(**kwargs): return _xresnet1d(4, [3, 4, 23, 3], **kwargs)
def xresnet1d152(**kwargs): return _xresnet1d(4, [3, 8, 36, 3], **kwargs)
def xresnet1d18_deep  (**kwargs): return _xresnet1d(1, [2,2,2,2,1,1], **kwargs)
def xresnet1d34_deep  (**kwargs): return _xresnet1d(1, [3,4,6,3,1,1], **kwargs)
def xresnet1d50_deep  (**kwargs): return _xresnet1d(4, [3,4,6,3,1,1], **kwargs)
def xresnet1d18_deeper(**kwargs): return _xresnet1d(1, [2,2,1,1,1,1,1,1], **kwargs)
def xresnet1d34_deeper(**kwargs): return _xresnet1d(1, [3,4,6,3,1,1,1,1], **kwargs)
def xresnet1d50_deeper(**kwargs): return _xresnet1d(4, [3,4,6,3,1,1,1,1], **kwargs)

# -----------------
# ENTRENAMIENTO - ENSEMBLE DE 5 MODELOS
# -----------------
num_models = 5
ensemble_model_paths = []
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0, 3.0, 1.5, 0.8, 2.0], device=device))  # Ajustar según tus clases
epochs = 200
early_stop_patience = 20

for model_id in range(num_models):
    print(f"\n Entrenando modelo {model_id + 1}/{num_models}")
    model = xresnet1d101(input_channels=12, num_classes=5, act_head="sigmoid").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Model {model_id+1} - Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Model {model_id+1} | Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            model_path = os.path.join(outputfolder, f"best_model_{model_id}.pth")
            torch.save(model.state_dict(), model_path)
            print("Modelo mejorado y guardado.")
        else:
            epochs_without_improvement += 1
            print(f"No mejora. {epochs_without_improvement}/{early_stop_patience} sin mejora.")
            if epochs_without_improvement >= early_stop_patience:
                print("Early stopping.")
                break

    ensemble_model_paths.append(model_path)

#Visualizar curva de perdida
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Pérdida de Entrenamiento')
plt.plot(val_losses, label='Pérdida de Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Curva de Pérdida por Época')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outputfolder, 'curva_perdida.png'))  # También guarda la figura
plt.show()

#----------------
#EVALUACIÓN
#---------------
from sklearn.metrics import classification_report

# ----------------
# EVALUACIÓN - ENSEMBLE
# ----------------
val_dataset_eval = H5Dataset(X_all, Y, val_idx)
val_loader_eval = DataLoader(val_dataset_eval, batch_size=batch_size)

# Cargar modelos entrenados y obtener predicciones promediadas
models = []
for model_path in ensemble_model_paths:
    m = xresnet1d101(input_channels=12, num_classes=5, act_head="sigmoid").to(device)
    m.load_state_dict(torch.load(model_path))
    m.eval()
    models.append(m)


print(" Modelos cargados para el ensemble.")

# Predicciones
all_preds_ensemble = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(val_loader_eval, desc="Ensemble Prediction"):
        inputs = inputs.to(device)
        outputs = [torch.sigmoid(m(inputs)) for m in models]
        avg_output = torch.stack(outputs).mean(dim=0)
        all_preds_ensemble.append(avg_output.cpu().numpy())
        all_labels.append(labels.numpy())

# Concatenar
all_preds = np.concatenate(all_preds_ensemble)
all_labels = np.concatenate(all_labels)

# Calcular thresholds óptimos
from sklearn.metrics import f1_score

def optimize_thresholds(y_true, y_pred_proba):
    thresholds = {}
    for i, class_name in enumerate(mlb.classes_):
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

optimal_thresholds = optimize_thresholds(all_labels, all_preds)
print("Optimal thresholds:", optimal_thresholds)

# Aplicar thresholds
preds_bin_opt = np.zeros_like(all_preds)
for i, class_name in enumerate(['CD', 'HYP', 'MI', 'NORM', 'STTC']):
    preds_bin_opt[:, i] = (all_preds[:, i] > optimal_thresholds[class_name]).astype(int)

# Reporte final
print(classification_report(all_labels, preds_bin_opt, target_names=mlb.classes_))

torch.save(model.state_dict(), os.path.join(outputfolder, 'inception_model.pth'))

# Guardar y_true y y_probs (val)
np.save(os.path.join(outputfolder, 'y_true_val.npy'), all_labels)
np.save(os.path.join(outputfolder, 'y_probs_val.npy'), all_preds)
print(" Guardados: y_true_val.npy y y_probs_val.npy")

#----------------------------------
# CURVA PR
#---------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import os

# Ruta de salida
outputfolder = r'C:\Users\maric\Desktop\TFM\output'

# Cargar arrays guardados
y_true = np.load(os.path.join(outputfolder, 'y_true_val.npy'))
y_probs = np.load(os.path.join(outputfolder, 'y_probs_val.npy'))

# Nombres de clases (ajústalo si has cambiado el orden en mlb.classes_)
class_names = ['CD', 'HYP', 'MI', 'NORM', 'STTC']

# Calcular micro promedio
precision_micro, recall_micro, _ = precision_recall_curve(y_true.ravel(), y_probs.ravel())
ap_micro = average_precision_score(y_true, y_probs, average="micro")

# Curva PR por clase
plt.figure(figsize=(10, 7))
plt.plot(recall_micro, precision_micro, label=f"Micro-average (AP = {ap_micro:.3f})", color='black', linewidth=2)

aps = []
for i, class_name in enumerate(class_names):
    precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
    ap = average_precision_score(y_true[:, i], y_probs[:, i])
    aps.append(ap)
    plt.plot(recall, precision, label=f"{class_name} (AP = {ap:.3f})")

# Macro promedio (media de APs)
ap_macro = np.mean(aps)
print(f"AP macro: {ap_macro:.3f}")
print(f"AP micro: {ap_micro:.3f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curvas Precisión-Recall (Validación)")
plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()

# Guardar gráfico
plt.savefig(os.path.join(outputfolder, 'curva_PR_val.png'))
print(" Curva PR guardada como 'curva_PR_val.png'")

plt.show()
