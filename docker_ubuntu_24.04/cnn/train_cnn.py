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
DATA_PATH = next((os.path.join(root, 'intel_lab_data_cleaned.csv') for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), '../data')) if 'intel_lab_data_cleaned.csv' in files), None) or exit("Erro: Arquivo 'intel_lab_data_cleaned.csv' não encontrado.")
OUTPUT_DIR = 'ml_results_cnn'
N_EPOCHS = 100
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else 'client1'
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definir CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=2, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 1, 32)
        self.fc2 = nn.Linear(32, 1)

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

# Função de treinamento e avaliação
def train_and_evaluate(X_train, X_test, y_train, y_test, client_id):
    start_time = time.time()
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                                  torch.tensor(y_train.values, dtype=torch.float32).to(device))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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
        
        # Avaliar RMSE
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
    
    # Gráfico de dispersão
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Real Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title(f'Predictions vs Actual - CNN ({client_id})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'cnn_scatter_{client_id}.png'))
    plt.close()
    
    # Gráfico de convergência
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_EPOCHS + 1), train_rmse, label='Train RMSE')
    plt.plot(range(1, N_EPOCHS + 1), test_rmse, label='Test RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title(f'Convergence - CNN ({client_id})')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'cnn_convergence_{client_id}.png'))
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
        json.dump(results, f)
    
    # Salvar modelo
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_{client_id}.pth'))
    
    return results

# Carregar e dividir dados
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Erro: Arquivo '{DATA_PATH}' não encontrado.")
    exit(1)

np.random.seed(42)
client_data = df.sample(frac=1.0 / 3.0, random_state=int(CLIENT_ID[-1])).reset_index(drop=True)
X = client_data[['moteid', 'humidity', 'light', 'voltage']]
y = client_data['temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar e salvar resultados
total_start_time = time.time()
results = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test, CLIENT_ID)
total_time = time.time() - total_start_time

# Salvar tempo total
with open(os.path.join(OUTPUT_DIR, f'total_time_{CLIENT_ID}.txt'), 'w') as f:
    f.write(f"Total Execution Time ({CLIENT_ID}): {total_time:.2f} seconds")