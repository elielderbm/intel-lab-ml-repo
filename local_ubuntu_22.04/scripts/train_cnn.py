import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    (os.path.join(root, 'intel_lab_data_cleaned.csv')
     for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), '../data'))
     if 'intel_lab_data_cleaned.csv' in files),
    None
) or exit("Erro: Arquivo 'intel_lab_data_cleaned.csv' não encontrado.")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../ml_results_cnn')
N_EPOCHS = 100
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN Melhorada
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.4)
        self.flatten_dim = 128 * 1
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

# Treinamento
def train_and_evaluate(X_train, X_test, y_train, y_test, client_id, model):
    print("[INFO] Iniciando treinamento...")
    start_time = time.time()

    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_train.values, dtype=torch.float32).to(device)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    epoch_times = []
    train_rmse = []
    test_rmse = []

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            train_pred = model(torch.tensor(X_train, dtype=torch.float32).to(device)).squeeze()
            test_pred = model(X_test_tensor).squeeze()
            train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred.cpu().numpy())))
            test_rmse.append(np.sqrt(mean_squared_error(y_test, test_pred.cpu().numpy())))

        epoch_times.append(time.time() - epoch_start)

    training_time = time.time() - start_time
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze().cpu().numpy()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[RESULTADO] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Real Temperature (\u00b0C)')
    plt.ylabel('Predicted Temperature (\u00b0C)')
    plt.title(f'Predictions vs Actual - CNN ({client_id})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'cnn_scatter_{client_id}.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_EPOCHS + 1), train_rmse, label='Train RMSE')
    plt.plot(range(1, N_EPOCHS + 1), test_rmse, label='Test RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title(f'Convergence - CNN ({client_id})')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'cnn_convergence_{client_id}.png'))
    plt.close()

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

        model = CNN().to(device)
        coeffs_path = os.path.join(OUTPUT_DIR, f'coeffs_{CLIENT_ID}.pt')
        if os.path.exists(coeffs_path):
            model.load_state_dict(torch.load(coeffs_path))
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

        torch.save(model.state_dict(), coeffs_path)

        with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
            f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds\n")

        time.sleep(1800)

    except Exception as e:
        print(f"[ERRO] Ocorreu um erro: {e}")
        print("[PROCESSO] Tentando novamente em 30 minutos...\n")
