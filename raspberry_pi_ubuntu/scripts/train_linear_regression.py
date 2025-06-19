import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../ml_results_linear_regression')
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Função de treinamento e avaliação
def train_and_evaluate(X_train, X_test, y_train, y_test, client_id, feature_names, prev_coefs=None, prev_intercept=None, prev_alpha=None):
    start_time = time.time()
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=alphas, cv=5))
    ])

    # Ajustar pipeline normalmente
    pipeline.fit(X_train, y_train)

    # Se houver coeficientes anteriores, inicializa o modelo com eles antes de treinar
    if prev_coefs is not None and prev_intercept is not None:
        pipeline.named_steps['ridge'].coef_ = prev_coefs
        pipeline.named_steps['ridge'].intercept_ = prev_intercept
        if prev_alpha is not None:
            pipeline.named_steps['ridge'].alpha_ = prev_alpha
        # Re-treina para ajustar a partir dos coeficientes anteriores
        pipeline.fit(X_train, y_train)
        print("[INFO] Coeficientes anteriores carregados e modelo atualizado.")

    else:
        print("[INFO] Modelo criado e treinado do zero.")

    training_time = time.time() - start_time

    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    cross_val_r2 = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='r2')
    cross_val_rmse = np.sqrt(-cross_val_score(pipeline, X_train, y_train, cv=3, scoring='neg_mean_squared_error'))

    poly = pipeline.named_steps['poly']
    expanded_feature_names = poly.get_feature_names_out(feature_names)

    coefs = pipeline.named_steps['ridge'].coef_
    intercept = pipeline.named_steps['ridge'].intercept_
    alpha = pipeline.named_steps['ridge'].alpha_

    results = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'training_time': float(training_time),
        'cross_val_r2_mean': float(cross_val_r2.mean()),
        'cross_val_rmse_mean': float(cross_val_rmse.mean()),
        'best_alpha': float(alpha),
        'coefs': dict(zip(expanded_feature_names, coefs.astype(float))),
        'intercept': float(intercept),
        'y_mean': float(np.mean(y_test)),
        'y_std': float(np.std(y_test)),
        'pred_mean': float(np.mean(y_pred)),
        'pred_std': float(np.std(y_pred))
    }

    with open(os.path.join(OUTPUT_DIR, f'results_{client_id}.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Salvar apenas coeficientes, intercept e alpha em arquivo separado
    coefs_dict = {
        'coefs': coefs.tolist(),
        'intercept': float(intercept),
        'alpha': float(alpha)
    }
    with open(os.path.join(OUTPUT_DIR, f'coeffs_{client_id}.json'), 'w') as f:
        json.dump(coefs_dict, f, indent=4)

    # Salvar modelo completo
    with open(os.path.join(OUTPUT_DIR, f'model_{client_id}.pkl'), 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"[RESULTADO] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    print("[INFO] Resultados, coeficientes e modelo salvos com sucesso.\n")

    return results, pipeline


#######################################
# Loop Infinito - Main
#######################################
while True:
    print("\n[PROCESSO] Iniciando novo ciclo de treinamento e validação...\n")

    try:
        df = pd.read_csv(DATA_PATH)

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        np.random.seed(42)
        client_data = df.sample(frac=0.03, random_state=int(CLIENT_ID[-1])).reset_index(drop=True)

        feature_names = ['humidity', 'light']
        if 'voltage' in df.columns:
            feature_names.append('voltage')

        X = client_data[feature_names]
        y = client_data['temperature']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Carregar apenas coeficientes anteriores, se existirem
        coeffs_path = os.path.join(OUTPUT_DIR, f'coeffs_{CLIENT_ID}.json')
        prev_coefs = None
        prev_intercept = None
        prev_alpha = None
        if os.path.exists(coeffs_path):
            with open(coeffs_path, 'r') as f:
                coeffs_data = json.load(f)
                prev_coefs = np.array(coeffs_data['coefs'])
                prev_intercept = coeffs_data['intercept']
                prev_alpha = coeffs_data.get('alpha')
            print("[INFO] Coeficientes anteriores carregados. Continuando o treinamento...")
        else:
            print("[INFO] Nenhum coeficiente anterior encontrado. Criando novo modelo...")

        # Treinar e avaliar
        total_start_time = time.time()
        results, pipeline = train_and_evaluate(
            X_train, X_test, y_train, y_test, CLIENT_ID, feature_names, prev_coefs, prev_intercept, prev_alpha
        )
        total_time = time.time() - total_start_time

        print(f"[INFO] Tempo total da execução: {total_time:.2f} segundos")
        print("[PROCESSO] Aguardando 30 minutos para próxima execução...\n")

        with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
            f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds\n")

        time.sleep(1800)  # Esperar 30 minutos

    except Exception as e:
        print(f"[ERRO] Ocorreu um erro: {e}")
        print("[PROCESSO] Tentando novamente em 30 minutos...\n")
        time.sleep(1800)
