import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pickle
import time
import os
import sys
import json
import traceback
from scipy.stats import loguniform

DATA_PATH = next(
    (os.path.join(root, 'beijing_pm2_dataset.csv')
     for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), '../data'))
     if 'beijing_pm2_dataset.csv' in files),
    None
) or exit("Erro: Arquivo 'beijing_pm2_dataset.csv' não encontrado.")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../ml_results_linear_regression')
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_and_evaluate(X_train, X_test, y_train, y_test, client_id, feature_names,
                       prev_coef=None, prev_intercept=None,
                       alpha=0.0001, learning_rate='invscaling', eta0=0.01):
    print(f"[INFO] Treinando com alpha={alpha:.6g}, learning_rate={learning_rate}, eta0={eta0:.6g}")

    model = SGDRegressor(
        penalty='l2',
        alpha=alpha,
        max_iter=10000,
        tol=1e-5,
        warm_start=True,
        random_state=42,
        learning_rate=learning_rate,
        eta0=eta0
    )

    if prev_coef is not None and prev_intercept is not None:
        prev_coef = np.array(prev_coef).flatten()
        model.coef_ = prev_coef
        model.intercept_ = np.array([prev_intercept])
        model._initialized = True

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[RESULTADO] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    return {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'training_time': training_time,
        'coef': model.coef_,
        'intercept': model.intercept_[0],
        'alpha': alpha,
        'learning_rate': learning_rate,
        'eta0': eta0,
        'y_pred': y_pred,
        'y_test': y_test
    }


def save_model_and_results(model, coef, intercept, params, client_id, feature_names, X_test, y_test):
    with open(os.path.join(OUTPUT_DIR, f'coeffs_{client_id}.json'), 'w') as f:
        json.dump({'coefs': coef.tolist(), 'intercept': float(intercept), 'alpha': params['alpha'],
                   'learning_rate': params['learning_rate'], 'eta0': params['eta0']}, f, indent=4)

    with open(os.path.join(OUTPUT_DIR, f'model_{client_id}.pkl'), 'wb') as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Real Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title(f'Predictions vs Actual - SGDRegressor ({client_id})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'prediction_scatter_{client_id}.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.xlabel("Prediction Error (°C)")
    plt.ylabel("Frequency")
    plt.title(f"Residuals Histogram - SGDRegressor ({client_id})")
    plt.savefig(os.path.join(OUTPUT_DIR, f"residuals_histogram_{client_id}.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, coef)
    plt.xlabel('Coefficient Value')
    plt.title(f'Feature Influence (SGDRegressor) - {client_id}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'coefficients_{client_id}.png'))
    plt.close()

    print("[INFO] Melhor modelo e gráficos salvos.")


def progressive_training_loop(X_train, X_test, y_train, y_test, client_id, feature_names,
                              prev_coef=None, prev_intercept=None):
    best_score = -np.inf
    best_params = {
        'alpha': 0.0001,
        'learning_rate': 'invscaling',
        'eta0': 0.01
    }
    best_model = None

    history_path = os.path.join(OUTPUT_DIR, f'history_{client_id}.json')
    # Limpa histórico antigo
    if os.path.exists(history_path):
        os.remove(history_path)

    for ciclo in range(10):
        alpha = loguniform(1e-5, 1e-1).rvs()
        eta0 = loguniform(1e-4, 1e-1).rvs()
        learning_rate = np.random.choice(['invscaling', 'adaptive', 'constant'])

        # Perturbação mínima para evitar overfitting estático
        X_train_perturbed = X_train + np.random.normal(0, 1e-5, X_train.shape)

        results = train_and_evaluate(
            X_train_perturbed, X_test, y_train, y_test, client_id, feature_names,
            prev_coef, prev_intercept,
            alpha=alpha,
            learning_rate=learning_rate,
            eta0=eta0
        )

        # Debug coeficientes
        coef_info = dict(zip(feature_names, results['coef']))
        print("[DEBUG] Coeficientes:", coef_info)

        # Salva histórico do ciclo
        with open(history_path, 'a') as f:
            json.dump({
                'cycle': ciclo + 1,
                'alpha': alpha,
                'eta0': eta0,
                'learning_rate': learning_rate,
                'r2': results['r2'],
                'rmse': results['rmse'],
                'mae': results['mae']
            }, f)
            f.write('\n')

        if results['r2'] > best_score:
            best_score = results['r2']
            best_params = {
                'alpha': alpha,
                'learning_rate': learning_rate,
                'eta0': eta0
            }
            best_model = results['model']
            best_coef = results['coef']
            best_intercept = results['intercept']
            print("[MELHORIA] Novo melhor modelo encontrado.")
        else:
            print("[INFO] Nenhuma melhoria neste ciclo.")

        print(f"[CICLO {ciclo+1}/10] Melhor R2: {best_score:.4f} com params: {best_params}")
        time.sleep(1)

    save_model_and_results(best_model, best_coef, best_intercept, best_params,
                           client_id, feature_names, X_test, y_test)

    return best_params, best_model, best_coef, best_intercept


while True:
    print("\n[PROCESSO] Iniciando novo ciclo de treinamento e validação...\n")
    try:
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=['TEMP'])

        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])

        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek

        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Features originais
        features = ['pm2.5', 'DEWP', 'PRES', 'Iws', 'sin_hour', 'cos_hour', 'dayofweek']

        # Features de interação (novas)
        df['pm2.5_x_Iws'] = df['pm2.5'] * df['Iws']
        df['DEWP_x_PRES'] = df['DEWP'] * df['PRES']
        features.extend(['pm2.5_x_Iws', 'DEWP_x_PRES'])

        target = 'TEMP'

        df = df[features + [target]]

        # Filtro de outliers via z-score (limite 4 desvios)
        z_scores = np.abs((df[features] - df[features].mean()) / df[features].std())
        df = df[(z_scores < 4).all(axis=1)]

        imputer = SimpleImputer(strategy='median')
        df[features] = imputer.fit_transform(df[features])

        seed = 42 + int(CLIENT_ID[-1])
        client_data = df.sample(frac=1.0 / 3.0, random_state=seed).reset_index(drop=True)

        X = client_data[features]
        y = client_data[target]

        numeric_features = features
        preprocessor = ColumnTransformer([
            ('num', RobustScaler(), numeric_features)
        ])

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        for train_idx, test_idx in splitter.split(X, client_data['dayofweek']):
            X_train_raw = X.iloc[train_idx]
            X_test_raw = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)
        feature_names = numeric_features

        coeffs_path = os.path.join(OUTPUT_DIR, f'coeffs_{CLIENT_ID}.json')
        prev_coef = prev_intercept = None
        if os.path.exists(coeffs_path):
            with open(coeffs_path, 'r') as f:
                data = json.load(f)
                prev_coef = np.array(data['coefs'])
                prev_intercept = data['intercept']
            print("[INFO] Coeficientes anteriores carregados.")
        else:
            print("[INFO] Nenhum coeficiente anterior encontrado.")

        total_start_time = time.time()
        progressive_training_loop(X_train, X_test, y_train, y_test, CLIENT_ID, feature_names,
                                  prev_coef, prev_intercept)
        total_time = time.time() - total_start_time

        print(f"[INFO] Tempo total da execução: {total_time:.2f} segundos")
        print("[PROCESSO] Aguardando 30 minutos para próxima execução...\n")

        with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
            f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds\n")

        time.sleep(1800)

    except Exception as e:
        print(f"[ERRO] Ocorreu um erro: {e}")
        print(traceback.format_exc())
        print("[PROCESSO] Tentando novamente em 30 minutos...\n")
        time.sleep(1800)
