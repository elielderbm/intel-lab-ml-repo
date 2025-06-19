import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import sys
import json

# Configuração
DATA_PATH = next((os.path.join(root, 'intel_lab_data_cleaned.csv')
                   for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), '../data'))
                   if 'intel_lab_data_cleaned.csv' in files), None) or exit("Erro: Arquivo 'intel_lab_data_cleaned.csv' não encontrado.")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../ml_results_cnn')
N_EPOCHS = 20
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cpu')


# Definir CNN simplificada
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=2, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(4 * 1, 8)
        self.fc2 = nn.Linear(8, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Função de treinamento e avaliação incremental
def train_and_evaluate(X_train, X_test, y_train, y_test, client_id, model=None):
    start_time = time.time()

    if model is None:
        model = CNN().to(device)
        print("[INFO] Modelo novo criado.")
    else:
        print("[INFO] Coeficientes anteriores carregados para continuar treinamento.")

    optimizer = optim.Adam(model.parameters(), lr=0.0007)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_train.values, dtype=torch.float32).to(device)
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

    best_loss = float('inf')
    patience = 2
    patience_counter = 0

    for epoch in range(N_EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Early stopping check
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor).squeeze()
            val_loss = criterion(val_outputs, y_test_tensor).item()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze().cpu().numpy()

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    results = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'training_time': training_time
    }

    with open(os.path.join(OUTPUT_DIR, f'results_{client_id}.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Salvar apenas coeficientes (state_dict)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'coeffs_{client_id}.pt'))

    # Salvar modelo completo para referência
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_{client_id}.pth'))

    print(f"[RESULTADO] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    print("[INFO] Resultados e coeficientes salvos.\n")

    return results, model


##########################################
# Loop Infinito
##########################################
while True:
    print("\n[PROCESSO] Novo ciclo de treinamento CNN iniciado...\n")

    try:
        df = pd.read_csv(DATA_PATH)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        np.random.seed(42)
        client_data = df.sample(frac=0.03, random_state=int(CLIENT_ID[-1])).reset_index(drop=True)

        X = client_data[['humidity', 'light']]
        y = client_data['temperature']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)

        # Carregar apenas coeficientes anteriores, se existirem
        coeffs_path = os.path.join(OUTPUT_DIR, f'coeffs_{CLIENT_ID}.pt')
        model = None
        if os.path.exists(coeffs_path):
            model = CNN().to(device)
            model.load_state_dict(torch.load(coeffs_path, map_location=device))
            print("[INFO] Coeficientes anteriores carregados.")
        else:
            print("[INFO] Nenhum coeficiente anterior encontrado. Criando novo modelo...")

        total_start_time = time.time()
        results, model = train_and_evaluate(
            X_train_scaled, X_test_scaled, y_train, y_test, CLIENT_ID, model
        )
        total_time = time.time() - total_start_time

        print(f"[INFO] Tempo total da execução: {total_time:.2f} segundos")
        print("[PROCESSO] Aguardando 30 minutos para a próxima execução...\n")

        with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
            f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds\n")

        time.sleep(1800)  # 30 minutos

    except Exception as e:
        print(f"[ERRO] Ocorreu um erro: {e}")
        print("[PROCESSO] Tentando novamente em 30 minutos...\n")
