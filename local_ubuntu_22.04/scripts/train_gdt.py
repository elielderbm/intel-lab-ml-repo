import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../ml_results_xgb')
N_ITERATIONS = 100
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Função para criar e configurar o modelo XGBoost
def create_xgb_model():
    return xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

# Treinamento
def train_and_evaluate(X_train, X_test, y_train, y_test, client_id, model):
    print("[INFO] Iniciando treinamento...")
    start_time = time.time()

    iteration_times = []
    train_rmse = []
    test_rmse = []

    # Simulando épocas com boosting iterations
    for iteration in range(N_ITERATIONS):
        iteration_start = time.time()
        
        # Treinamento incremental
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )

        # Avaliação
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))
        test_rmse.append(np.sqrt(mean_squared_error(y_test, test_pred)))

        iteration_times.append(time.time() - iteration_start)

    training_time = time.time() - start_time
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[RESULTADO] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Real Temperature (\u00b0C)')
    plt.ylabel('Predicted Temperature (\u00b0C)')
    plt.title(f'Predictions vs Actual - XGBoost ({client_id})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'xgb_scatter_{client_id}.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_ITERATIONS + 1), train_rmse, label='Train RMSE')
    plt.plot(range(1, N_ITERATIONS + 1), test_rmse, label='Test RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title(f'Convergence - XGBoost ({client_id})')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'xgb_convergence_{client_id}.png'))
    plt.close()

    results = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'training_time': training_time,
        'iteration_times': iteration_times,
        'avg_iteration_time': np.mean(iteration_times),
        'y_pred': y_pred.tolist(),
        'y_test': y_test.tolist(),
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }

    with open(os.path.join(OUTPUT_DIR, f'results_{client_id}.json'), 'w') as f:
        json.dump(results, f, indent=4)

    model.save_model(os.path.join(OUTPUT_DIR, f'model_{client_id}.json'))

    print("[INFO] Resultados e modelo salvos com sucesso.\n")

    return results

#######################################
# Loop Infinito - Main
#######################################
while True:
    print("\n[PROCESSO] Iniciando novo ciclo de treinamento e validação...\n")

    try:
        df = pd.read_csv(DATA_PATH)
        df['temperature'] = df['temperature'].apply(lambda x: x if 0 <= x <= 100 else np.nan)
        df = df.dropna().reset_index(drop=True)

        np.random.seed(42)
        client_data = df.sample(frac=1.0 / 3.0, random_state=int(CLIENT_ID[-1])).reset_index(drop=True)

        # Análise de correlação e remoção de variáveis altamente correlacionadas
        corr = client_data[['humidity', 'light', 'voltage']].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        selected_features = [col for col in ['humidity', 'light', 'voltage'] if col not in to_drop]

        X = client_data[selected_features]
        y = client_data['temperature']

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

        model = create_xgb_model()
        coeffs_path = os.path.join(OUTPUT_DIR, f'coeffs_{CLIENT_ID}.json')
        if os.path.exists(coeffs_path):
            model.load_model(coeffs_path)
            print("[INFO] Coeficientes anteriores carregados. Continuando o treinamento...")
        else:
            print("[INFO] Nenhum coeficiente anterior encontrado. Criando novo modelo...")

        total_start_time = time.time()
        results = train_and_evaluate(
            X_train_scaled, X_test_scaled, pd.Series(y_train_scaled), pd.Series(y_test_scaled), CLIENT_ID, model
        )
        total_time = time.time() - total_start_time

        y_pred = scaler_y.inverse_transform(np.array(results['y_pred']).reshape(-1, 1)).flatten()
        y_test = scaler_y.inverse_transform(np.array(results['y_test']).reshape(-1, 1)).flatten()
        results['y_pred'] = y_pred.tolist()
        results['y_test'] = y_test.tolist()
        results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        results['mae'] = mean_absolute_error(y_test, y_pred)
        results['r2'] = r2_score(y_test, y_pred)

        with open(os.path.join(OUTPUT_DIR, f'results_{CLIENT_ID}.json'), 'w') as f:
            json.dump(results, f, indent=4)

        print(f"[INFO] Tempo total da execução: {total_time:.2f} segundos")
        print("[PROCESSO] Aguardando 30 minutos para próxima execução...\n")

        model.save_model(coeffs_path)

        with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
            f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds\n")

        time.sleep(1800)

    except Exception as e:
        print(f"[ERRO] Ocorreu um erro: {e}")
        print("[PROCESSO] Tentando novamente em 30 minutos...\n")