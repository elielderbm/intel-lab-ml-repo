import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train, y_train)
    # Se houver coeficientes anteriores, inicializa o modelo com eles antes de treinar
    if prev_coefs is not None and prev_intercept is not None:
        model.coef_ = prev_coefs
        model.intercept_ = prev_intercept
        # Re-treina para ajustar a partir dos coeficientes anteriores
        model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[RESULTADO] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, Alpha: {model.alpha_}")

    # Gráfico de dispersão
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Real Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title(f'Predictions vs Actual - RidgeCV ({client_id})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'prediction_scatter_{client_id}.png'))
    plt.close()

    # Histograma dos resíduos
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.xlabel("Prediction Error (°C)")
    plt.ylabel("Frequency")
    plt.title(f"Residuals Histogram - RidgeCV ({client_id})")
    plt.savefig(os.path.join(OUTPUT_DIR, f"residuals_histogram_{client_id}.png"))
    plt.close()

    # Distribuição dos reais vs previstos
    plt.figure(figsize=(8, 6))
    plt.hist(y_test, bins=50, alpha=0.5, label='Actual')
    plt.hist(y_pred, bins=50, alpha=0.5, label='Predicted')
    plt.title(f'Distribution Comparison - {client_id}')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'hist_actual_vs_pred_{client_id}.png'))
    plt.close()

    # Erro por faixa de temperatura
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

    # Coeficientes
    coef = model.coef_
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, coef)
    plt.xlabel('Coefficient Value')
    plt.title(f'Feature Influence (Ridge) - {client_id}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'coefficients_{client_id}.png'))
    plt.close()

    # Cross-validation
    cross_val_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cross_val_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))

    # Salvar resultados
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

    # Salvar apenas coeficientes e intercept em arquivo separado
    coefs_dict = {
        'coefs': coef.tolist(),
        'intercept': float(model.intercept_)
    }
    with open(os.path.join(OUTPUT_DIR, f'coeffs_{client_id}.json'), 'w') as f:
        json.dump(coefs_dict, f, indent=4)

    # Salvar modelo completo
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

        np.random.seed(42)
        client_data = df.sample(frac=1.0 / 3.0, random_state=int(CLIENT_ID[-1])).reset_index(drop=True)

        # Separar X e y
        X = client_data[['moteid', 'humidity', 'light', 'voltage']]
        y = client_data['temperature']

        # Pré-processamento
        numeric_features = ['humidity', 'light', 'voltage']
        categorical_features = ['moteid']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
            ]
        )

        X_processed = preprocessor.fit_transform(X)
        feature_names = (
            numeric_features +
            list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
        )

        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        # Carregar apenas coeficientes se existirem
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

        # Treinar e avaliar
        total_start_time = time.time()
        results = train_and_evaluate(X_train, X_test, y_train, y_test, CLIENT_ID, feature_names, prev_coefs, prev_intercept)
        total_time = time.time() - total_start_time

        print(f"[INFO] Tempo total da execução: {total_time:.2f} segundos")
        print("[PROCESSO] Aguardando 30 minutos para próxima execução...\n")

        with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
            f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds\n")

        time.sleep(1800)  # 30 minutos

    except Exception as e:
        print(f"[ERRO] Ocorreu um erro: {e}")
        print("[PROCESSO] Tentando novamente em 30 minutos...\n")
        time.sleep(1800)
