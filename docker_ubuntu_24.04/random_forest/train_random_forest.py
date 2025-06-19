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
import gc

# Configuração
DATA_PATH = next(
    (os.path.join(root, 'intel_lab_data_cleaned.csv')
     for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), '../data'))
     if 'intel_lab_data_cleaned.csv' in files),
    None
) or exit("Erro: Arquivo 'intel_lab_data_cleaned.csv' não encontrado.")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../ml_results_random_forest')
N_ESTIMATORS = 100  # Reduzido para menor uso de recurso e memória
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Função de treinamento e avaliação
def train_and_evaluate(X_train, X_test, y_train, y_test, client_id, prev_params=None):
    start_time = time.time()

    if prev_params:
        print("[INFO] Parâmetros anteriores carregados. Continuando o treinamento...")
        model = RandomForestRegressor(
            n_estimators=prev_params['n_estimators'] + N_ESTIMATORS,
            random_state=42,
            max_depth=prev_params.get('max_depth', 16),  # Limita profundidade para reduzir memória
            min_samples_split=prev_params.get('min_samples_split', 2),
            min_samples_leaf=prev_params.get('min_samples_leaf', 2),  # Aumenta folhas mínimas
            max_features=prev_params.get('max_features', 'auto'),
            bootstrap=prev_params.get('bootstrap', True),
            n_jobs=2  # Limita a apenas 1 núcleo para reduzir uso de CPU/memória
        )
        model.fit(X_train, y_train)
    else:
        print("[INFO] Nenhum parâmetro anterior encontrado. Criando novo modelo...")
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=42,
            max_depth=16,  # Limita profundidade para reduzir memória
            min_samples_leaf=2,  # Aumenta folhas mínimas
            n_jobs=2
        )
        model.fit(X_train, y_train)

    training_time = time.time() - start_time
    avg_tree_time = training_time / model.n_estimators

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
    plt.clf()
    plt.close('all')

    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [f'X{i}' for i in indices], rotation=45)
    plt.title(f'Feature Importance - Random Forest ({client_id})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'random_forest_importance_{client_id}.png'))
    plt.close()
    plt.clf()
    plt.close('all')

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
        'n_estimators': model.n_estimators
    }
    with open(os.path.join(OUTPUT_DIR, f'results_{client_id}.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Salvar apenas parâmetros essenciais do modelo (não o modelo completo)
    params_dict = {
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'min_samples_split': model.min_samples_split,
        'min_samples_leaf': model.min_samples_leaf,
        'max_features': model.max_features,
        'bootstrap': model.bootstrap
    }
    with open(os.path.join(OUTPUT_DIR, f'params_{client_id}.json'), 'w') as f:
        json.dump(params_dict, f, indent=4)

    # Salvar modelo completo
    with open(os.path.join(OUTPUT_DIR, f'model_{client_id}.pkl'), 'wb') as f:
        pickle.dump(model, f)

    # Libera memória explicitamente
    del model, y_pred, importances, indices
    gc.collect()

    return results

############################
# Loop Infinito - Main
############################
while True:
    print("\n[PROCESSO] Iniciando novo ciclo de treinamento e validação...\n")

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{DATA_PATH}' não encontrado.")
        exit(1)

    # Preparação dos dados
    np.random.seed(42)
    client_data = df.sample(frac=1.0 / 3.0, random_state=int(CLIENT_ID[-1])).reset_index(drop=True)
    X = client_data[['moteid', 'humidity', 'light', 'voltage']]
    y = client_data['temperature']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Carregar apenas parâmetros essenciais se existirem
    params_path = os.path.join(OUTPUT_DIR, f'params_{CLIENT_ID}.json')
    prev_params = None
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            prev_params = json.load(f)
        print("[INFO] Parâmetros anteriores carregados. Continuando o treinamento...")
    else:
        print("[INFO] Nenhum parâmetro anterior encontrado. Criando novo modelo...")

    # Treinar e avaliar
    total_start_time = time.time()
    results = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test, CLIENT_ID, prev_params)
    total_time = time.time() - total_start_time

    print(f"[RESULTADO] RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}, R2: {results['r2']:.4f}")
    print(f"[INFO] Número total de árvores no modelo atual: {results['n_estimators']}")
    print(f"[INFO] Tempo total da execução: {total_time:.2f} segundos")
    print("[PROCESSO] Aguardando 15 minutos para próxima execução...\n")

    with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
        f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds\n")

    # Libera memória explicitamente
    del df, client_data, X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, results
    gc.collect()

    time.sleep(900)  # Esperar 15 minutos
