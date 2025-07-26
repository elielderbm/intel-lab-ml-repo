import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle
import time
import os
import sys
import json
import gc
import xgboost as xgb

# Configuração
DATA_PATH = next(
    (os.path.join(root, 'weather_hourly_clean.csv')
     for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), '../data'))
     if 'weather_hourly_clean.csv' in files),
    None
) or exit("Erro: Arquivo 'weather_hourly_clean.csv' não encontrado.")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../ml_results_xgboost')
N_ESTIMATORS = 100  # Quantidade de árvores adicionadas por ciclo
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'

os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_RESULT_PATH = os.path.join(OUTPUT_DIR, f'best_rmse_{CLIENT_ID}.txt')

def train_and_evaluate(X_train, X_test, y_train, y_test, client_id, feature_names, prev_params=None):
    start_time = time.time()

    if prev_params:
        print("[INFO] Parâmetros anteriores carregados. Continuando o treinamento...")
        n_estimators = prev_params.get('n_estimators', 100) + N_ESTIMATORS
    else:
        print("[INFO] Nenhum parâmetro anterior encontrado. Criando novo modelo...")
        n_estimators = 100

    model_path = os.path.join(OUTPUT_DIR, f'model_{client_id}.json')
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=8,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=84,
        verbosity=1,
        tree_method='hist',
        predictor='cpu_predictor',
        enable_categorical=False,
        use_label_encoder=False
    )

    if os.path.exists(model_path) and prev_params:
        model.load_model(model_path)
        model.n_estimators = n_estimators
        model.fit(X_train, y_train, xgb_model=model_path)
    else:
        model.fit(X_train, y_train)

    training_time = time.time() - start_time
    avg_tree_time = training_time / n_estimators

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Temperatura Real (°C)')
    plt.ylabel('Temperatura Prevista (°C)')
    plt.title(f'Previsão vs Real - XGBoost ({client_id})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'xgb_scatter_{client_id}.png'))
    plt.close()

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title(f'Importância das Features - XGBoost ({client_id})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'xgb_importance_{client_id}.png'))
    plt.close()

    # Gráfico do erro (Real - Previsto)
    plt.figure(figsize=(10, 4))
    plt.plot(np.array(y_test) - np.array(y_pred), label='Erro (Real - Previsto)', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Erro de Previsão - XGBoost ({client_id})')
    plt.xlabel('Amostras')
    plt.ylabel('Erro (°C)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'xgb_error_{client_id}.png'))
    plt.close()

    results = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'training_time': training_time,
        'avg_tree_time': avg_tree_time,
        'y_pred': y_pred.tolist(),
        'y_test': y_test.tolist(),
        'feature_importances': importances.tolist(),
        'n_estimators': n_estimators
    }

    # Salva modelo só se RMSE melhorou
    try:
        with open(BEST_RESULT_PATH) as f:
            best_rmse = float(f.read())
    except FileNotFoundError:
        best_rmse = float('inf')

    if rmse < best_rmse:
        model.save_model(model_path)
        with open(BEST_RESULT_PATH, 'w') as f:
            f.write(str(rmse))
        print("[INFO] Novo modelo salvo. RMSE melhorado.")
    else:
        print("[INFO] RMSE não melhorou. Modelo anterior mantido.")

    with open(os.path.join(OUTPUT_DIR, f'results_{client_id}.json'), 'w') as f:
        json.dump(results, f, indent=4)

    params_dict = {
        'n_estimators': n_estimators,
        'max_depth': 8,
        'learning_rate': 0.05,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    with open(os.path.join(OUTPUT_DIR, f'params_{client_id}.json'), 'w') as f:
        json.dump(params_dict, f, indent=4)

    del model, y_pred, importances, indices
    gc.collect()

    return results

# Loop principal
while True:
    print("\n[PROCESSO] Iniciando novo ciclo de treinamento e validação...\n")

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{DATA_PATH}' não encontrado.")
        exit(1)

    np.random.seed(84)
    client_data = df.sample(frac=1.0 / 3.0, random_state=int(CLIENT_ID[-1])).reset_index(drop=True)

    client_data['datetime'] = pd.to_datetime(client_data['datetime'])
    client_data['hour'] = client_data['datetime'].dt.hour
    client_data['day'] = client_data['datetime'].dt.day
    client_data['month'] = client_data['datetime'].dt.month
    client_data['weekday'] = client_data['datetime'].dt.weekday
    client_data['day_of_year'] = client_data['datetime'].dt.dayofyear
    client_data['is_weekend'] = client_data['weekday'].apply(lambda x: int(x >= 5))
    client_data['season'] = ((client_data['month'] % 12 + 3) // 3).astype(int)
    client_data['hour_bin'] = pd.cut(client_data['hour'], bins=[-1, 5, 11, 17, 23], labels=['madrugada', 'manhã', 'tarde', 'noite'])

    client_data['temperature'] = client_data['temperature'] - 273.15
    client_data['temperature'] = client_data['temperature'].clip(-30, 50)
    client_data['humidity'] = client_data['humidity'].clip(0, 100)
    client_data['pressure'] = client_data['pressure'].clip(900, 1100)
    client_data['wind_speed'] = client_data['wind_speed'].clip(0, 50)

    client_data['sin_hour'] = np.sin(2 * np.pi * client_data['hour'] / 24)
    client_data['cos_hour'] = np.cos(2 * np.pi * client_data['hour'] / 24)
    client_data['wind_direction'] = np.radians(client_data['wind_direction'] % 360)
    client_data['wind_x'] = np.cos(client_data['wind_direction']) * client_data['wind_speed']
    client_data['wind_y'] = np.sin(client_data['wind_direction']) * client_data['wind_speed']

    # Novas features adicionadas
    client_data['is_night'] = client_data['hour'].apply(lambda h: int(h < 6 or h > 18))
    client_data['is_working_hour'] = client_data['hour'].apply(lambda h: int(9 <= h <= 17))

    client_data.sort_values('datetime', inplace=True)
    client_data['temp_lag1'] = client_data['temperature'].shift(1)
    client_data['temp_lag1'].fillna(method='bfill', inplace=True)

    X_raw = client_data[[
        'city', 'humidity', 'pressure', 'weather_desc', 'wind_speed', 'wind_x', 'wind_y',
        'sin_hour', 'cos_hour', 'day', 'month', 'weekday', 'day_of_year',
        'season', 'is_weekend', 'hour_bin', 'is_night', 'is_working_hour', 'temp_lag1'
    ]]
    y = client_data['temperature']

    numeric_features = ['humidity', 'pressure', 'wind_speed', 'wind_x', 'wind_y',
                        'sin_hour', 'cos_hour', 'day', 'month', 'weekday',
                        'day_of_year', 'season', 'is_weekend', 'is_night',
                        'is_working_hour', 'temp_lag1']
    categorical_features = ['city', 'weather_desc', 'hour_bin']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

    X_processed = preprocessor.fit_transform(X_raw)
    feature_names = (
        numeric_features +
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=84
    )

    params_path = os.path.join(OUTPUT_DIR, f'params_{CLIENT_ID}.json')
    prev_params = None
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            prev_params = json.load(f)
        print("[INFO] Parâmetros anteriores carregados.")
    else:
        print("[INFO] Nenhum parâmetro anterior encontrado. Modelo será criado do zero.")

    total_start_time = time.time()
    results = train_and_evaluate(X_train, X_test, y_train, y_test, CLIENT_ID, feature_names, prev_params)
    total_time = time.time() - total_start_time

    print(f"[RESULTADO] RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}, R2: {results['r2']:.4f}")
    print(f"[INFO] Número total de árvores no modelo atual: {results['n_estimators']}")
    print(f"[INFO] Tempo total da execução: {total_time:.2f} segundos")
    print("[PROCESSO] Aguardando 15 minutos para próxima execução...\n")

    with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
        f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds\n")

    del df, client_data, X_raw, y, X_train, X_test, y_train, y_test, X_processed, results
    gc.collect()

    time.sleep(900)
