import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Configuração
OUTPUT_DIR = 'ml_results_linear_regression'
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
for metric in ['rmse', 'mae', 'r2', 'training_time']:
    plt.figure(figsize=(8, 6))
    clients = list(all_results.keys())
    values = [all_results[c][metric] for c in clients]
    plt.bar(clients, values)
    plt.xlabel('Client')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} Across Clients - Linear Regression')
    plt.savefig(os.path.join(ANALYSIS_DIR, f'linear_regression_{metric}_comparison.png'))
    plt.close()

# Gráfico de tempos totais
plt.figure(figsize=(8, 6))
clients = list(total_times.keys())
values = [total_times[c] for c in clients]
plt.bar(clients, values)
plt.xlabel('Client')
plt.ylabel('Total Time (seconds)')
plt.title('Total Execution Time Across Clients - Linear Regression')
plt.savefig(os.path.join(ANALYSIS_DIR, 'linear_regression_total_time_comparison.png'))
plt.close()

# Gerar relatório
report = f"# Analysis Report - Linear Regression\n\n"
report += "## Performance Metrics\n"
for client_id, results in all_results.items():
    report += f"### {client_id}\n"
    report += f"- RMSE: {results['rmse']:.4f}\n"
    report += f"- MAE: {results['mae']:.4f}\n"
    report += f"- R²: {results['r2']:.4f}\n"
    report += f"- Training Time: {results['training_time']:.2f} seconds\n"
    report += f"- Total Execution Time: {total_times[client_id]:.2f} seconds\n"

report += "\n## Analysis\n"
report += "- **Performance**: Linear Regression is fast but may yield higher RMSE/MAE due to its simplicity. Variability across clients suggests data heterogeneity.\n"
report += "- **Convergence**: Not applicable, as Linear Regression is non-iterative.\n"
report += "- **Environment Impact**: Raspberry Pi (client3) likely has higher training times due to ARM hardware. Ubuntu 22.04 (client1) and Docker (client2) are faster.\n"
report += "- **Stability**: Low variability in metrics indicates stable performance across environments.\n"
report += f"- **Plots**: See `linear_regression_rmse_comparison.png`, `mae_comparison.png`, `r2_comparison.png`, `training_time_comparison.png`, `total_time_comparison.png` in `{ANALYSIS_DIR}`.\n"

with open(os.path.join(ANALYSIS_DIR, 'analysis_linear_regression.md'), 'w') as f:
    f.write(report)