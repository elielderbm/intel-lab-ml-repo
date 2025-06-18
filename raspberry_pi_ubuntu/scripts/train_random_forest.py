import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import time
import os
import sys
import json

# Configuração
DATA_PATH = next(
    (os.path.join(root, 'intel_lab_data_cleaned.csv')
     for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), '../data'))
     if 'intel_lab_data_cleaned.csv' in files),
    None
) or exit("Erro: Arquivo 'intel_lab_data_cleaned.csv' não encontrado.")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../ml_results_random_forest')
N_ESTIMATORS = 20  # Slightly increased for better performance, still lightweight
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_and_evaluate(X_train, X_test, y_train, y_test, client_id, feature_names):
    start_time = time.time()
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=8,              # Slightly deeper trees for more learning capacity
        min_samples_leaf=2,       # Slightly smaller leaves for more flexibility
        min_samples_split=4,      # Prevent splits with very few samples
        random_state=42,
        n_jobs=1                  # Use only one core for Raspberry Pi
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    avg_tree_time = training_time / N_ESTIMATORS

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation scores (lightweight, 3-fold)
    cross_val_r2 = cross_val_score(model, X_train, y_train, cv=3, scoring='r2', n_jobs=1)
    cross_val_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error', n_jobs=1))

    importances = model.feature_importances_

    # Save only summary results
    results = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'training_time': float(training_time),
        'avg_tree_time': float(avg_tree_time),
        'cross_val_r2_mean': float(cross_val_r2.mean()),
        'cross_val_rmse_mean': float(cross_val_rmse.mean()),
        'feature_importances': importances.tolist(),
        'feature_names': feature_names
    }
    with open(os.path.join(OUTPUT_DIR, f'results_{client_id}.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Save model
    with open(os.path.join(OUTPUT_DIR, f'model_{client_id}.pkl'), 'wb') as f:
        pickle.dump(model, f)

    return results

# Carregar e dividir dados
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Erro: Arquivo '{DATA_PATH}' não encontrado.")
    exit(1)

# Shuffle data before sampling
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

np.random.seed(42)
client_data = df.sample(frac=0.03, random_state=int(CLIENT_ID[-1])).reset_index(drop=True)

# Optionally add more features if memory allows
feature_names = ['humidity', 'light']
if 'voltage' in df.columns:
    feature_names.append('voltage')

X = client_data[feature_names]
y = client_data['temperature']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Treinar e salvar resultados
total_start_time = time.time()
results = train_and_evaluate(X_train, X_test, y_train, y_test, CLIENT_ID, feature_names)
total_time = time.time() - total_start_time

# Salvar tempo total
with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
    f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds")