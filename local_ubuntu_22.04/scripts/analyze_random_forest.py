import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Diretórios
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../ml_results_random_forest')
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

# Gráficos por client
def plot_client_analysis(client_id, results):
    y_test = np.array(results['y_test'])
    y_pred = np.array(results['y_pred'])
    residuals = y_test - y_pred

    plt.figure(figsize=(18, 5))

    # Predito vs Real com regressão
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs Actual')
    m, b = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test, m*y_test + b, 'r--', label=f'Fit y={m:.2f}x+{b:.2f}')
    plt.xlabel('Actual Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title(f'Client {client_id} - Predicted vs Actual')
    plt.legend()
    plt.grid(True)

    # Distribuição dos resíduos
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)
    plt.title(f'Client {client_id} - Residuals Distribution')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Resíduos vs Previsões
    plt.subplot(1, 3, 3)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='dashed')
    plt.xlabel('Predicted Temperature (°C)')
    plt.ylabel('Residual')
    plt.title(f'Client {client_id} - Residuals vs Predicted')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, f'client_{client_id}_detailed_analysis.png'))
    plt.close()

    # Importância das features
    importances = np.array(results['feature_importances'])
    feature_names = results['feature_names']
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title(f'Feature Importance - Random Forest ({client_id})')
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, f'client_{client_id}_feature_importance.png'))
    plt.close()

# Gerar gráficos detalhados por client
for client_id, results in all_results.items():
    plot_client_analysis(client_id, results)

# Comparações entre clientes
metrics = ['rmse', 'mae', 'r2', 'training_time']
for metric in metrics:
    plt.figure(figsize=(8, 6))
    clients = list(all_results.keys())
    values = [all_results[c][metric] for c in clients]
    plt.bar(clients, values, color='skyblue', edgecolor='black')
    plt.xlabel('Client')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} Across Clients - Random Forest')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(ANALYSIS_DIR, f'random_forest_{metric}_comparison.png'))
    plt.close()

# Tempo total de execução
plt.figure(figsize=(8, 6))
clients = list(total_times.keys())
values = [total_times[c] for c in clients]
plt.bar(clients, values, color='lightgreen', edgecolor='black')
plt.xlabel('Client')
plt.ylabel('Total Time (seconds)')
plt.title('Total Execution Time Across Clients - Random Forest')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(ANALYSIS_DIR, 'random_forest_total_time_comparison.png'))
plt.close()

# Relatório com recomendações
report = f"# Analysis Report - Random Forest\n\n"
report += "## Performance Metrics per Client\n"
for client_id, results in all_results.items():
    report += f"### Client: {client_id}\n"
    report += f"- RMSE: {results['rmse']:.4f}\n"
    report += f"- MAE: {results['mae']:.4f}\n"
    report += f"- R²: {results['r2']:.4f}\n"
    report += f"- Training Time: {results['training_time']:.2f} seconds\n"
    report += f"- Total Execution Time: {total_times[client_id]:.2f} seconds\n\n"

report += "## Visual Analysis\n"
report += ("Each client has visualizations that include:\n"
           "- Predicted vs Actual values with regression line\n"
           "- Histogram of residuals to assess prediction bias\n"
           "- Residuals vs Predictions to detect model limitations\n"
           "- Feature importance bar chart to understand model focus\n\n")

report += "## Insights and Recommendations\n"
report += "- **Model Quality**: Random Forest provides strong performance for structured data and captures nonlinearities well.\n"
report += "- **Residuals**: Should be symmetrically distributed around zero. Asymmetry may indicate bias or missing predictors.\n"
report += "- **Feature Importances**: Features with very low importance could be candidates for removal or require better encoding.\n"
report += "- **Number of Trees (`n_estimators`)**: Higher values improve stability but increase training time. Use 100–500 in practice.\n"
report += "- **Depth and Leaf Parameters**: Tune `max_depth`, `min_samples_split`, and `min_samples_leaf` to control overfitting.\n"
report += "- **Scaling**: Although not required for trees, consistent scaling is helpful for preprocessing and hybrid models.\n"
report += "- **Training Time**: Monitor for clients with slow training — may indicate data imbalance or unnecessary complexity.\n"
report += "- **Outliers**: Strong outliers may bias predictions. Investigate data quality or apply outlier mitigation.\n"
report += "- **Client Differences**: Large metric variability across clients suggests need for localized or federated modeling.\n\n"

report += "## Next Steps\n"
report += "- Use GridSearchCV or RandomizedSearchCV to optimize hyperparameters.\n"
report += "- Consider feature engineering or automated feature selection.\n"
report += "- Test on unseen clients to validate generalization.\n"
report += "- Combine with ensemble or boosting models for comparison.\n"

# Salvar relatório
report_path = os.path.join(ANALYSIS_DIR, 'analysis_random_forest.md')
with open(report_path, 'w') as f:
    f.write(report)

"Análise finalizada e relatório gerado com sucesso."