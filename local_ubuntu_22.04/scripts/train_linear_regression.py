import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
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

# Função de treinamento e avaliação
def train_and_evaluate(X_train, X_test, y_train, y_test, client_id, feature_names, prev_coefs=None, prev_intercept=None):
    print("[INFO] Iniciando treinamento...")

    start_time = time.time()
    alphas = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0]  # Valores intermediários para balancear regularização
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train, y_train)
    if prev_coefs is not None and prev_intercept is not None:
        model.coef_ = prev_coefs
        model.intercept_ = prev_intercept
        model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[RESULTADO] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, Alpha: {model.alpha_}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Real Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title(f'Predictions vs Actual - RidgeCV ({client_id})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'prediction_scatter_{client_id}.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.xlabel("Prediction Error (°C)")
    plt.ylabel("Frequency")
    plt.title(f"Residuals Histogram - RidgeCV ({client_id})")
    plt.savefig(os.path.join(OUTPUT_DIR, f"residuals_histogram_{client_id}.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(y_test, bins=50, alpha=0.5, label='Actual')
    plt.hist(y_pred, bins=50, alpha=0.5, label='Predicted')
    plt.title(f'Distribution Comparison - {client_id}')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'hist_actual_vs_pred_{client_id}.png'))
    plt.close()

    df_eval = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
    df_eval['temp_bin'] = pd.cut(df_eval['y_true'], bins=10)
    grouped = df_eval.groupby('temp_bin').apply(
        lambda g: pd.Series({
            'mae': mean_absolute_error(g['y_true'], g['y_pred']),
            'rmse': np.sqrt(mean_squared_error(g['y_true'], g['y_pred']))
        })
    )
    grouped.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Error by Temperature Range - {client_id}')
    plt.ylabel('Error')
    plt.xlabel('Temperature Bins')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'error_by_temp_bin_{client_id}.png'))
    plt.close()

    coef = model.coef_
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, coef)
    plt.xlabel('Coefficient Value')
    plt.title(f'Feature Influence (Ridge) - {client_id}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'coefficients_{client_id}.png'))
    plt.close()

    cross_val_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cross_val_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))

    results = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'training_time': training_time,
        'cross_val_r2_mean': cross_val_r2.mean(),
        'cross_val_rmse_mean': cross_val_rmse.mean(),
        'best_alpha': model.alpha_,
        'coefs': dict(zip(feature_names, coef)),
        'intercept': float(model.intercept_),
        'residuals': residuals.tolist(),
        'y_pred': y_pred.tolist(),
        'y_test': y_test.tolist(),
        'y_mean': np.mean(y_test),
        'y_std': np.std(y_test),
        'pred_mean': np.mean(y_pred),
        'pred_std': np.std(y_pred)
    }

    with open(os.path.join(OUTPUT_DIR, f'results_{client_id}.json'), 'w') as f:
        json.dump(results, f, indent=4)

    coefs_dict = {
        'coefs': coef.tolist(),
        'intercept': float(model.intercept_)
    }
    with open(os.path.join(OUTPUT_DIR, f'coeffs_{client_id}.json'), 'w') as f:
        json.dump(coefs_dict, f, indent=4)

    with open(os.path.join(OUTPUT_DIR, f'model_{client_id}.pkl'), 'wb') as f:
        pickle.dump(model, f)

    print("[INFO] Resultados, coeficientes e modelo salvos com sucesso.\n")

    return results

#######################################
# Loop Infinito - Main
#######################################
while True:
    print("\n[PROCESSO] Iniciando novo ciclo de treinamento e validação...\n")

    try:
        df = pd.read_csv(DATA_PATH)

        # 🔍 Limpeza de outliers - Ajustado para temperatura
        df = df[
            (df['temperature'] < 45) & (df['temperature'] > 5) &  # Mais restritivo
            (df['humidity'] < 100) & (df['humidity'] > 10) &
            (df['voltage'] < 3.5) & (df['voltage'] > 2.0)
        ].copy()

        # np.random.seed e amostragem - Aumentada para 65%
        np.random.seed(int(time.time()) % 1000)
        client_data = df.sample(frac=0.65, random_state=int(CLIENT_ID[-1])).reset_index(drop=True)

        # 📈 FEATURES OTIMIZADAS
        client_data['humidity_squared'] = client_data['humidity'] ** 2
        client_data['log_light'] = np.log1p(client_data['light'] + 1)
        client_data['voltage_per_light'] = client_data['voltage'] / (client_data['light'] + 1)
        client_data['humidity_light_interaction'] = client_data['humidity'] * client_data['light']
        client_data['temp_diff'] = client_data['temperature'].diff().fillna(0)
        client_data['humidity_voltage_ratio'] = client_data['humidity'] / (client_data['voltage'] + 1)
        client_data['light_voltage_interaction'] = client_data['light'] * client_data['voltage']
        client_data['epoch_normalized'] = (client_data['epoch'] - client_data['epoch'].min()) / (client_data['epoch'].max() - client_data['epoch'].min())  # Nova feature temporal

        # Seleção final de variáveis
        feature_cols = ['humidity', 'light', 'voltage', 'humidity_squared',
                        'log_light', 'voltage_per_light', 'humidity_light_interaction',
                        'temp_diff', 'humidity_voltage_ratio', 'light_voltage_interaction',
                        'epoch_normalized']
        X = client_data[feature_cols]
        y = client_data['temperature']

        # Escalonamento
        preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), feature_cols)]
        )

        X_processed = preprocessor.fit_transform(X)
        feature_names = feature_cols  # já está na ordem correta

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        coeffs_path = os.path.join(OUTPUT_DIR, f'coeffs_{CLIENT_ID}.json')
        prev_coefs = None
        prev_intercept = None
        if os.path.exists(coeffs_path):
            with open(coeffs_path, 'r') as f:
                coeffs_data = json.load(f)
                prev_coefs = np.array(coeffs_data['coefs'])
                prev_intercept = coeffs_data['intercept']
            print("[INFO] Coeficientes anteriores carregados. Continuando o treinamento...")
        else:
            print("[INFO] Nenhum coeficiente anterior encontrado. Criando novo modelo.")

        total_start_time = time.time()
        results = train_and_evaluate(X_train, X_test, y_train, y_test, CLIENT_ID, feature_names, prev_coefs, prev_intercept)
        total_time = time.time() - total_start_time

        print(f"[INFO] Tempo total da execução: {total_time:.2f} segundos")
        print("[PROCESSO] Aguardando 30 sec para próxima execução...\n")

        with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
            f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds\n")

        time.sleep(30)

    except Exception as e:
        print(f"[ERRO] Ocorreu um erro: {e}")
        print("[PROCESSO] Tentando novamente em 30 sec...\n")
        time.sleep(30)