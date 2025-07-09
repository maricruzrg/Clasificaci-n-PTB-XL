# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:17:35 2025
Para cargar luego cargar el archivo:
ecg_signals = np.load('ptbxl_all_ecg_signals.npy', allow_pickle=True)

"""
#Extraer señales de todos los registros y guardarla en un unico npy
import pandas as pd
import wfdb
import numpy as np
import os
from tqdm import tqdm  

# Ruta base del dataset
BASE_PATH = r'C:\Users\maric\Desktop\ptbxl\PTBXL'
RECORD_BASE = os.path.join(BASE_PATH, 'records100')  # 'records500' también es usado, lo manejaremos abajo

# Leer el CSV con los metadatos
df = pd.read_csv(os.path.join(BASE_PATH, 'ptbxl_database.csv'))

# Lista para guardar las señales
all_signals = []

# Recorremos todas las filas con barra de progreso
for _, row in tqdm(df.iterrows(), total=len(df)):
    record_path = os.path.join(BASE_PATH, row['filename_lr'])  # archivo .hea/.dat de baja resolución
    try:
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal  # array de forma (1000, 12)
        all_signals.append(signal)
    except Exception as e:
        print(f"Error leyendo {record_path}: {e}")

# Guardar todas las señales en un único archivo .npy
np.save('ptbxl_all_ecg_signals.npy', all_signals)

"""
Ahora extraer las señales individuales vinculadas con su scp_codes
"""
import ast

# Ruta base de PTB-XL
BASE_PATH = r'C:\Users\maric\Desktop\ptbxl\PTBXL'
SIGNAL_OUTPUT_DIR = 'ecg_signals_individuales'
os.makedirs(SIGNAL_OUTPUT_DIR, exist_ok=True)

# Cargar el archivo CSV
df = pd.read_csv(os.path.join(BASE_PATH, 'ptbxl_database.csv'))

# Convertir columna de string a diccionario real
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

# Guardar etiquetas y rutas
labels_list = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    record_path = os.path.join(BASE_PATH, row['filename_lr'])  # archivo .hea/.dat
    ecg_id = row['ecg_id']
    signal_filename = f'ecg_{ecg_id}.npy'
    signal_path = os.path.join(SIGNAL_OUTPUT_DIR, signal_filename)
    
    try:
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal  # (1000, 12)
        np.save(signal_path, signal)

        # Convertimos los scp_codes en una sola cadena de etiquetas (puedes cambiar esto)
        labels = list(row['scp_codes'].keys())
        labels_list.append({'ecg_id': ecg_id, 'filename': signal_filename, 'labels': ';'.join(labels)})
    
    except Exception as e:
        print(f"Error con {record_path}: {e}")

# Guardamos las etiquetas
labels_df = pd.DataFrame(labels_list)
labels_df.to_csv('labels.csv', index=False)
