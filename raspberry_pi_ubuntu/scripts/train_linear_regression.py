import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../ml_results_linear_regression')
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_and_evaluate(X_train, X_test, y_train, y_test, client_id, feature_names):
    start_time = time.time()
    alphas = [0.1, 1.0, 10.0]  # Try a small range of alphas
    model = RidgeCV(alphas=alphas, cv=3)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation scores
    cross_val_r2 = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
    cross_val_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error'))

    # Save only summary results
    results = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'training_time': float(training_time),
        'cross_val_r2_mean': float(cross_val_r2.mean()),
        'cross_val_rmse_mean': float(cross_val_rmse.mean()),
        'best_alpha': float(model.alpha_),
        'coefs': dict(zip(feature_names, model.coef_.astype(float))),
        'y_mean': float(np.mean(y_test)),
        'y_std': float(np.std(y_test)),
        'pred_mean': float(np.mean(y_pred)),
        'pred_std': float(np.std(y_pred))
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

# Use only two features for minimal memory/CPU
X = client_data[['humidity', 'light']]
y = client_data['temperature']

# Pré-processamento com StandardScaler
numeric_features = ['humidity', 'light']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ]
)

X_processed = preprocessor.fit_transform(X).astype(np.float32)
feature_names = numeric_features

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Treinar e salvar resultados
total_start_time = time.time()
results = train_and_evaluate(X_train, X_test, y_train, y_test, CLIENT_ID, feature_names)
total_time = time.time() - total_start_time

# Salvar tempo total
with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
    f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds")
