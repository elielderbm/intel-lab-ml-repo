import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Configuração dos diretórios
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../ml_results_cnn')
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

#############################################
# Gráficos Comparativos de Métricas
#############################################
metrics = ['rmse', 'mae', 'r2', 'training_time', 'avg_epoch_time']
for metric in metrics:
    values = [all_results[c][metric] for c in clients]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(clients, values, color='steelblue', edgecolor='black')
    plt.xlabel('Client')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} Across Clients - CNN')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, f'cnn_{metric}_comparison.png'))
    plt.close()

#############################################
# Gráfico de Tempo Total
#############################################
values = [total_times[c] for c in clients]
plt.figure(figsize=(8, 6))
bars = plt.bar(clients, values, color='darkorange', edgecolor='black')
plt.xlabel('Client')
plt.ylabel('Total Time (seconds)')
plt.title('Total Execution Time Across Clients - CNN')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{val:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'cnn_total_time_comparison.png'))
plt.close()

#############################################
# Gráfico de Convergência (RMSE por Época)
#############################################
plt.figure(figsize=(10, 6))
for client_id, results in all_results.items():
    plt.plot(range(1, len(results['test_rmse']) + 1), results['test_rmse'],
             marker='o', label=f'{client_id} Test RMSE')

plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Convergence Across Clients - CNN')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'cnn_convergence_comparison.png'))
plt.close()

#############################################
# Relatório em Markdown
#############################################
report = "# CNN Analysis Report\n\n"
report += "## Performance Metrics per Client\n\n"

for client_id, results in all_results.items():
    report += f"### Client: {client_id}\n"
    report += f"- **RMSE:** {results['rmse']:.4f}\n"
    report += f"- **MAE:** {results['mae']:.4f}\n"
    report += f"- **R²:** {results['r2']:.4f}\n"
    report += f"- **Training Time:** {results['training_time']:.2f} seconds\n"
    report += f"- **Average Epoch Time:** {results['avg_epoch_time']:.4f} seconds\n"
    report += f"- **Total Execution Time:** {total_times[client_id]:.2f} seconds\n\n"

report += "## Visual and Statistical Analysis\n"
report += "- **RMSE & MAE:** Measure prediction error. Lower values indicate better performance.\n"
report += "- **R² Score:** Measures how well the model explains variance. Close to 1 is better.\n"
report += "- **Training & Epoch Time:** Important for evaluating hardware and efficiency.\n"
report += "- **Convergence:** Observed by plotting RMSE across epochs. Plateau or sudden increases indicate overfitting, underfitting, or poor hyperparameters.\n\n"

report += "## Insights & Recommendations\n"
report += "- **Check Data Normalization:** Especially important for CNN input consistency.\n"
report += "- **Learning Rate and Scheduler:** Consider learning rate decay if RMSE plateaus.\n"
report += "- **Model Complexity:** If underfitting, increase layers, filters, or use batch normalization and dropout.\n"
report += "- **Batch Size:** Adjust batch size to stabilize gradients and improve convergence.\n"
report += "- **Client Data Variability:** Consider FedAvg, FedProx, or personalized layers if clients' data distributions are heterogeneous.\n"
report += "- **Regularization:** Apply dropout and L2 regularization to prevent overfitting.\n"
report += "- **Data Augmentation:** If applicable to input type, it improves generalization.\n"
report += "- **Early Stopping:** Useful if convergence stalls or overfitting starts.\n\n"

report += "## Available Plots\n"
for metric in metrics:
    report += f"- `cnn_{metric}_comparison.png`\n"
report += "- `cnn_total_time_comparison.png`\n"
report += "- `cnn_convergence_comparison.png`\n\n"

report += "## Next Steps\n"
report += "- Experiment with different CNN architectures.\n"
report += "- Apply hyperparameter tuning (learning rate, batch size, optimizers).\n"
report += "- Evaluate federated personalization methods.\n"
report += "- Test robustness to data heterogeneity.\n"
report += "- Explore techniques like knowledge distillation for model compression.\n"

with open(os.path.join(ANALYSIS_DIR, 'analysis_cnn.md'), 'w') as f:
    f.write(report)
