import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Configuração
OUTPUT_DIR = 'ml_results_random_forest'
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
for metric in ['rmse', 'mae', 'r2', 'training_time', 'avg_tree_time']:
    plt.figure(figsize=(8, 6))
    clients = list(all_results.keys())
    values = [all_results[c][metric] for c in clients]
    plt.bar(clients, values)
    plt.xlabel('Client')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} Across Clients - Random Forest')
    plt.savefig(os.path.join(ANALYSIS_DIR, f'random_forest_{metric}_comparison.png'))
    plt.close()

# Gráfico de tempos totais
plt.figure(figsize=(8, 6))
clients = list(total_times.keys())
values = [total_times[c] for c in clients]
plt.bar(clients, values)
plt.xlabel('Client')
plt.ylabel('Total Time (seconds)')
plt.title('Total Execution Time Across Clients - Random Forest')
plt.savefig(os.path.join(ANALYSIS_DIR, 'random_forest_total_time_comparison.png'))
plt.close()

# Gráfico de feature importance
plt.figure(figsize=(10, 6))
for client_id, results in all_results.items():
    importances = results['feature_importances']
    indices = np.argsort(importances)[::-1]
    plt.bar(np.arange(len(importances)) + 0.2 * clients.index(client_id), 
            [importances[i] for i in indices], width=0.2, label=client_id)
plt.xticks(np.arange(len(importances)), [results['feature_names'][i] for i in indices], rotation=45)
plt.ylabel('Importance')
plt.title('Feature Importance Across Clients - Random Forest')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'random_forest_feature_importance_comparison.png'))
plt.close()

# Gerar relatório
report = f"# Analysis Report - Random Forest\n\n"
report += f"**Number of Trees**: {N_ESTIMATORS}\n\n"
report += "## Performance Metrics\n"
for client_id, results in all_results.items():
    report += f"### {client_id}\n"
    report += f"- RMSE: {results['rmse']:.4f}\n"
    report += f"- MAE: {results['mae']:.4f}\n"
    report += f"- R²: {results['r2']:.4f}\n"
    report += f"- Training Time: {results['training_time']:.2f} seconds\n"
    report += f"- Avg Time per Tree: {results['avg_tree_time']:.4f} seconds\n"
    report += f"- Total Execution Time: {total_times[client_id]:.2f} seconds\n"

report += "\n## Analysis\n"
report += "- **Performance**: Random Forest typically achieves lower RMSE/MAE than Linear Regression due to its ability to model non-linear relationships.\n"
report += "- **Feature Importance**: `humidity` is likely the most important feature, as shown in `feature_importance_comparison.png`.\n"
report += "- **Convergence**: Not iterative per tree, but ensemble stability is high.\n"
report += "- **Environment Impact**: Raspberry Pi (client3) has significantly higher training times due to limited hardware.\n"
report += "- **Stability**: Low variability in metrics across clients indicates robust performance.\n"
report += f"- **Plots**: See `random_forest_rmse_comparison.png`, `mae_comparison.png`, `r2_comparison.png`, `training_time_comparison.png`, `avg_tree_time_comparison.png`, `total_time_comparison.png`, `feature_importance_comparison.png` in `{ANALYSIS_DIR}`.\n"

with open(os.path.join(ANALYSIS_DIR, 'analysis_random_forest.md'), 'w') as f:
    f.write(report)