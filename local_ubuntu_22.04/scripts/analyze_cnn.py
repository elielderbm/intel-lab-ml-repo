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

clients = list(all_results.keys())

# Gráficos comparativos
metrics = ['rmse', 'mae', 'r2', 'training_time', 'avg_epoch_time']
for metric in metrics:
    values = [all_results[c][metric] for c in clients]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(clients, values, color='skyblue')
    plt.xlabel('Client')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} Across Clients - CNN')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, f'cnn_{metric}_comparison.png'))
    plt.close()

# Tempo total de execução
values = [total_times[c] for c in clients]
plt.figure(figsize=(8, 6))
bars = plt.bar(clients, values, color='salmon')
plt.xlabel('Client')
plt.ylabel('Total Time (seconds)')
plt.title('Total Execution Time Across Clients - CNN')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'cnn_total_time_comparison.png'))
plt.close()

# Convergência
plt.figure(figsize=(10, 6))
for client_id, results in all_results.items():
    plt.plot(range(1, len(results['test_rmse']) + 1), results['test_rmse'], label=f'{client_id} Test RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Convergence Across Clients - CNN')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'cnn_convergence_comparison.png'))
plt.close()

# Gerar relatório
report = f"# CNN Analysis Report\n\n"
report += "## Performance Metrics by Client\n"
for client_id, results in all_results.items():
    report += f"### {client_id}\n"
    report += f"- RMSE: {results['rmse']:.4f}\n"
    report += f"- MAE: {results['mae']:.4f}\n"
    report += f"- R²: {results['r2']:.4f}\n"
    report += f"- Training Time: {results['training_time']:.2f} seconds\n"
    report += f"- Avg Epoch Time: {results['avg_epoch_time']:.4f} seconds\n"
    report += f"- Total Time: {total_times[client_id]:.2f} seconds\n"

report += "\n## Visual & Quantitative Analysis\n"
report += "- **RMSE & MAE**: Indicate prediction error magnitude. Lower is better.\n"
report += "- **R² Score**: Measures variance explained by the model. Close to 1 is ideal.\n"
report += "- **Training & Epoch Times**: Useful to assess hardware efficiency.\n"
report += "- **Convergence**: Sudden plateau or increase in RMSE may suggest overfitting or poor learning rate.\n"

report += "\n## Recommendations for Improvement\n"
report += "- **Data Normalization**: Verify consistency across clients.\n"
report += "- **Learning Rate & Epochs**: Tune learning rate and increase epochs gradually while monitoring RMSE.\n"
report += "- **Model Complexity**: Consider adding layers, dropout, or batch norm if underfitting.\n"
report += "- **Batch Size**: Tune batch size to stabilize gradients.\n"
report += "- **Client-Specific Optimization**: Use FedAvg or personalization layers if data distribution differs significantly.\n"

report += f"\n## Plots Available\n"
for metric in metrics:
    report += f"- `cnn_{metric}_comparison.png`\n"
report += "- `cnn_total_time_comparison.png`\n"
report += "- `cnn_convergence_comparison.png`\n"

with open(os.path.join(ANALYSIS_DIR, 'analysis_cnn.md'), 'w') as f:
    f.write(report)
