import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import sys
import json

# Configuração
DATA_PATH = next(
    (os.path.join(root, 'weather_hourly_clean.csv')
     for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), '../data'))
     if 'weather_hourly_clean.csv' in files),
    None
) or exit("Erro: Arquivo 'weather_hourly_clean.csv' não encontrado.")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../ml_results_cnn')
N_EPOCHS = 200  # Aumentado para permitir mais treinamento
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definir MLP
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

# Função de treinamento e avaliação
def train_and_evaluate(X_train, X_test, y_train, y_test, client_id, model, y_scaler):
    print("[INFO] Iniciando treinamento...")
    start_time = time.time()

    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Learning rate reduzido
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_scaler.transform(y_train.values.reshape(-1, 1)), dtype=torch.float32).to(device)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    epoch_times = []
    train_rmse = []
    test_rmse = []
    best_test_rmse = float('inf')
    best_model_state = None
    patience = 20  # Early stopping patience
    counter = 0

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_scaler.transform(y_test.values.reshape(-1, 1)), dtype=torch.float32).to(device)

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Avaliação do RMSE
        model.eval()
        with torch.no_grad():
            train_pred = model(torch.tensor(X_train, dtype=torch.float32).to(device)).squeeze()
            test_pred = model(X_test_tensor).squeeze()
            # Reverter escalonamento para calcular métricas
            train_pred_rescaled = y_scaler.inverse_transform(train_pred.cpu().numpy().reshape(-1, 1)).flatten()
            test_pred_rescaled = y_scaler.inverse_transform(test_pred.cpu().numpy().reshape(-1, 1)).flatten()
            y_train_rescaled = y_train.values
            y_test_rescaled = y_test.values
            train_rmse.append(np.sqrt(mean_squared_error(y_train_rescaled, train_pred_rescaled)))
            test_rmse.append(np.sqrt(mean_squared_error(y_test_rescaled, test_pred_rescaled)))

        # Atualizar learning rate
        scheduler.step(total_loss / len(train_loader))
        epoch_times.append(time.time() - epoch_start)

        # Early stopping
        if test_rmse[-1] < best_test_rmse:
            best_test_rmse = test_rmse[-1]
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f"[INFO] Early stopping acionado na época {epoch+1}")
            break

    # Carregar melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    training_time = time.time() - start_time
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).squeeze().cpu().numpy()
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[RESULTADO] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # Gráfico de dispersão
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Real Temperature (C)')
    plt.ylabel('Predicted Temperature (C)')
    plt.title(f'Predictions vs Actual - MLP ({client_id})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'mlp_scatter_{client_id}.png'))
    plt.close()

    # Gráfico de convergência
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_rmse) + 1), train_rmse, label='Train RMSE')
    plt.plot(range(1, len(test_rmse) + 1), test_rmse, label='Test RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title(f'Convergence - MLP ({client_id})')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'mlp_convergence_{client_id}.png'))
    plt.close()

    # Salvar resultados
    results = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'training_time': training_time,
        'epoch_times': epoch_times,
        'avg_epoch_time': np.mean(epoch_times),
        'y_pred': y_pred.tolist(),
        'y_test': y_test.tolist(),
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }

    with open(os.path.join(OUTPUT_DIR, f'results_{client_id}.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Salvar modelo completo
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_{client_id}.pth'))

    print("[INFO] Resultados e modelo salvos com sucesso.\n")

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

        # Dados
        X_raw = client_data[['city', 'humidity', 'pressure', 'weather_desc', 'wind_direction', 'wind_speed']]
        y = client_data['temperature'] - 273.15  # Converter de Kelvin para Celsius

        # Pré-processamento: numéricas, categóricas e alvo
        numeric_features = ['humidity', 'pressure', 'wind_direction', 'wind_speed']
        categorical_features = ['city', 'weather_desc']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
            ]
        )

        X_processed = preprocessor.fit_transform(X_raw)
        input_size = X_processed.shape[1]

        # Escalonar o alvo (temperatura em Celsius)
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        # Instanciar modelo com input compatível
        model = MLP(input_size).to(device)

        coeffs_path = os.path.join(OUTPUT_DIR, f'coeffs_{CLIENT_ID}.pt')
        if os.path.exists(coeffs_path):
            model.load_state_dict(torch.load(coeffs_path))
            print("[INFO] Coeficientes anteriores carregados. Continuando o treinamento...")
        else:
            print("[INFO] Nenhum coeficiente anterior encontrado. Criando novo modelo...")

        # Treinar e avaliar
        total_start_time = time.time()
        results = train_and_evaluate(
            X_train, X_test, y_train, y_test, CLIENT_ID, model, y_scaler
        )
        total_time = time.time() - total_start_time

        print(f"[INFO] Tempo total da execução: {total_time:.2f} segundos")
        print("[PROCESSO] Aguardando 30 minutos para próxima execução...\n")

        # Salvar apenas coeficientes após o treinamento
        torch.save(model.state_dict(), coeffs_path)

        with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
            f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds\n")

        time.sleep(1800)  # 30 minutos

    except Exception as e:
        print(f"[ERRO] Ocorreu um erro: {e}")
        print("[PROCESSO] Tentando novamente em 30 minutos...\n")