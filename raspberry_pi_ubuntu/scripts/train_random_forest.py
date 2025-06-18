import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
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
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_and_evaluate(X_train, X_test, y_train, y_test, client_id, feature_names):
    start_time = time.time()

    # Pipeline: PolynomialFeatures + StandardScaler + RandomForestRegressor
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(
            n_estimators=16,      # Reduced number of trees
            max_depth=7,          # Shallower trees
            min_samples_leaf=2,
            min_samples_split=4,
            random_state=42,
            n_jobs=1              # Use only one core for Raspberry Pi
        ))
    ])

    # Further reduced hyperparameter grid for GridSearchCV
    param_grid = {
        'rf__max_depth': [6, 7],
        'rf__min_samples_leaf': [2, 4],
        'rf__min_samples_split': [2, 4],
        'rf__n_estimators': [12, 16]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    training_time = time.time() - start_time

    y_pred = best_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation scores (3-fold)
    cross_val_r2 = cross_val_score(best_pipeline, X_train, y_train, cv=3, scoring='r2', n_jobs=1)
    cross_val_rmse = np.sqrt(-cross_val_score(best_pipeline, X_train, y_train, cv=3, scoring='neg_mean_squared_error', n_jobs=1))

    # Get feature importances from the RF in the pipeline
    poly = best_pipeline.named_steps['poly']
    expanded_feature_names = poly.get_feature_names_out(feature_names)
    rf = best_pipeline.named_steps['rf']
    importances = rf.feature_importances_

    results = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'training_time': float(training_time),
        'cross_val_r2_mean': float(cross_val_r2.mean()),
        'cross_val_rmse_mean': float(cross_val_rmse.mean()),
        'best_params': grid_search.best_params_,
        'feature_importances': dict(zip(expanded_feature_names, importances)),
        'feature_names': expanded_feature_names.tolist(),
        'y_mean': float(np.mean(y_test)),
        'y_std': float(np.std(y_test)),
        'pred_mean': float(np.mean(y_pred)),
        'pred_std': float(np.std(y_pred))
    }
    with open(os.path.join(OUTPUT_DIR, f'results_{client_id}.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Save model pipeline
    with open(os.path.join(OUTPUT_DIR, f'model_{client_id}.pkl'), 'wb') as f:
        pickle.dump(best_pipeline, f)

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

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar e salvar resultados
total_start_time = time.time()
results = train_and_evaluate(X_train, X_test, y_train, y_test, CLIENT_ID, feature_names)
total_time = time.time() - total_start_time

# Salvar tempo total
with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
    f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds")