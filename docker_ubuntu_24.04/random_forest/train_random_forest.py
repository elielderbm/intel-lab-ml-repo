import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle
import time
import os
import sys
import json

# Configuração
DATA_PATH = next((os.path.join(root, 'intel_lab_data_cleaned.csv') for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), '../data')) if 'intel_lab_data_cleaned.csv' in files), None) or exit("Erro: Arquivo 'intel_lab_data_cleaned.csv' não encontrado.")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../ml_results_random_forest')
N_ESTIMATORS = 100
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Função de treinamento e avaliação
def train_and_evaluate(X_train, X_test, y_train, y_test, client_id):
    start_time = time.time()
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=42)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    avg_tree_time = training_time / N_ESTIMATORS
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Gráfico de dispersão
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Real Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title(f'Predictions vs Actual - Random Forest ({client_id})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'random_forest_scatter_{client_id}.png'))
    plt.close()
    
    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [X.columns[i] for i in indices], rotation=45)
    plt.title(f'Feature Importance - Random Forest ({client_id})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'random_forest_importance_{client_id}.png'))
    plt.close()
    
    # Salvar resultados
    results = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'training_time': training_time,
        'avg_tree_time': avg_tree_time,
        'y_pred': y_pred.tolist(),
        'y_test': y_test.tolist(),
        'feature_importances': importances.tolist(),
        'feature_names': X.columns.tolist()
    }
    with open(os.path.join(OUTPUT_DIR, f'results_{client_id}.json'), 'w') as f:
        json.dump(results, f)
    
    # Salvar modelo
    with open(os.path.join(OUTPUT_DIR, f'model_{client_id}.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    return results

# Carregar e dividir dados
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Erro: Arquivo '{DATA_PATH}' não encontrado.")
    exit(1)

np.random.seed(42)
client_data = df.sample(frac=1.0 / 3.0, random_state=int(CLIENT_ID[-1])).reset_index(drop=True)
X = client_data[['moteid', 'humidity', 'light', 'voltage']]
y = client_data['temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar e salvar resultados
total_start_time = time.time()
results = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test, CLIENT_ID)
total_time = time.time() - total_start_time

# Salvar tempo total
with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
    f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds")