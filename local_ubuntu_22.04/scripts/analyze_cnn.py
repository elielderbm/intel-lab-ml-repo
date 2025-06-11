import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Configuração
OUTPUT_DIR = 'ml_results_cnn'
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Carregar resultados
results_files = glob.glob(os.path.join(OUTPUT_DIR, 'results_client*.json'))
total_time_files = glob.glob(os.path.join(OUTPUT_DIR, 'total_time_client*.txt'))
all_results = {}
total_times = {}

for file in results_files:
    client_id = os.path.basename(file).split('_')[1].split('.')[0]
    with open(file, 'r') as f:
        all_results[client_id] = json.load(f)

for file in total_time_files:
    client_id = os.path.basename(file).split('_')[2].split('.')[0]
    with open(file, 'r') as f:
        total_times[client_id] = float(f.read().split(': ')[1].split(' ')[0])

# Gerar gráficos comparativos
for metric in ['rmse', 'mae', 'r2', 'training_time', 'avg_epoch_time']:
    plt.figure(figsize=(8, 6))
    clients = list(all_results.keys())
    values = [all_results[c][metric] for c in clients]
    plt.bar(clients, values)
    plt.xlabel('Client')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} Across Clients - CNN')
    plt.savefig(os.path.join(ANALYSIS_DIR, f'cnn_{metric}_comparison.png'))
    plt.close()

# Gráfico de tempos totais
plt.figure(figsize=(8, 6))
clients = list(total_times.keys())
values = [total_times[c] for c in clients]
plt.bar(clients, values)
plt.xlabel('Client')
plt.ylabel('Total Time (seconds)')
plt.title('Total Execution Time Across Clients - CNN')
plt.savefig(os.path.join(ANALYSIS_DIR, 'cnn_total_time_comparison.png'))
plt.close()

# Gráfico de convergência comparativo
plt.figure(figsize=(10, 6))
for client_id, results in all_results.items():
    plt.plot(range(1, len(results['test_rmse']) + 1), results['test_rmse'], label=f'{client_id} Test RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Convergence Across Clients - CNN')
plt.legend()
plt.savefig(os.path.join(ANALYSIS_DIR, 'cnn_convergence_comparison.png'))
plt.close()

# Gerar relatório
report = f"# Analysis Report - CNN\n\n"
report += f"**Number of Epochs**: {N_EPOCHS}\n\n"
report += "## Performance Metrics\n"
for client_id, results in all_results.items():
    report += f"### {client_id}\n"
    report += f"- RMSE: {results['rmse']:.4f}\n"
    report += f"- MAE: {results['mae']:.4f}\n"
    report += f"- R²: {results['r2']:.4f}\n"
    report += f"- Training Time: {results['training_time']:.2f} seconds\n"
    report += f"- Avg Time per Epoch: {results['avg_epoch_time']:.4f} seconds\n"
    report += f"- Total Execution Time: {total_times[client_id]:.2f} seconds\n"

report += "\n## Analysis\n"
report += "- **Performance**: CNN can achieve competitive RMSE/MAE but requires significant computational resources.\n"
report += "- **Convergence**: Test RMSE decreases over epochs, with potential overfitting if train/test RMSE diverge (see `cnn_convergence_comparison.png`).\n"
report += "- **Environment Impact**: Raspberry Pi (client3) has much higher training times due to ARM hardware limitations.\n"
report += "- **Stability**: Variability in RMSE across clients reflects data heterogeneity.\n"
report += f"- **Plots**: See `cnn_rmse_comparison.png`, `mae_comparison.png`, `r2_comparison.png`, `training_time_comparison.png`, `avg_epoch_time_comparison.png`, `total_time_comparison.png`, `cnn_convergence_comparison.png` in `{ANALYSIS_DIR}`.\n"

with open(os.path.join(ANALYSIS_DIR, 'analysis_cnn.md'), 'w') as f:
    f.write(report)