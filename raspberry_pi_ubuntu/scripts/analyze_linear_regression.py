import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Configuração
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../ml_results_linear_regression')
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

# Função para gráficos por client
def plot_client_analysis(client_id, results):
    y_test = np.array(results['y_test'])
    y_pred = np.array(results['y_pred'])
    residuals = y_test - y_pred

    plt.figure(figsize=(18, 5))

    # 1. Predicted vs Actual with regression line
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs Actual')
    m, b = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test, m*y_test + b, 'r--', label=f'Fit line y={m:.2f}x+{b:.2f}')
    plt.xlabel('Actual Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title(f'Client {client_id} - Predicted vs Actual')
    plt.legend()
    plt.grid(True)

    # 2. Residuals distribution
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)
    plt.title(f'Client {client_id} - Residuals Distribution')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True)

    # 3. Residuals vs Predicted values (to check for patterns)
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

# Gerar gráficos para cada client
for client_id, results in all_results.items():
    if 'y_test' not in results or 'y_pred' not in results:
        print(f"Warning: Skipping client {client_id} due to missing 'y_test' or 'y_pred' in results.")
        continue
    plot_client_analysis(client_id, results)

# Comparar métricas entre clients
metrics = ['rmse', 'mae', 'r2', 'training_time']
for metric in metrics:
    plt.figure(figsize=(8, 6))
    clients = list(all_results.keys())
    values = [all_results[c][metric] for c in clients if metric in all_results[c]]
    filtered_clients = [c for c in clients if metric in all_results[c]]
    plt.bar(filtered_clients, values, color='skyblue', edgecolor='black')
    plt.xlabel('Client')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} Across Clients - Linear Regression')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(ANALYSIS_DIR, f'linear_regression_{metric}_comparison.png'))
    plt.close()

# Gráfico de tempos totais
plt.figure(figsize=(8, 6))
clients = list(total_times.keys())
values = [total_times[c] for c in clients]
plt.bar(clients, values, color='lightgreen', edgecolor='black')
plt.xlabel('Client')
plt.ylabel('Total Time (seconds)')
plt.title('Total Execution Time Across Clients - Linear Regression')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(ANALYSIS_DIR, 'linear_regression_total_time_comparison.png'))
plt.close()

# Gerar relatório detalhado com explicações
report = f"# Analysis Report - Linear Regression\n\n"
report += "## Performance Metrics per Client\n"
for client_id, results in all_results.items():
    report += f"### Client: {client_id}\n"
    report += f"- RMSE: {results.get('rmse', float('nan')):.4f}\n"
    report += f"- MAE: {results.get('mae', float('nan')):.4f}\n"
    report += f"- R²: {results.get('r2', float('nan')):.4f}\n"
    report += f"- Training Time: {results.get('training_time', float('nan')):.2f} seconds\n"
    report += f"- Total Execution Time: {total_times.get(client_id, float('nan')):.2f} seconds\n\n"

report += "## Visual Analysis\n"
report += ("For each client, detailed plots were generated showing:\n"
           "- Predicted vs Actual values with a regression fit line to assess prediction accuracy.\n"
           "- Distribution of residuals (errors) to detect bias or skewness.\n"
           "- Residuals plotted against predicted values to check for patterns indicating model issues.\n\n")

report += "## Insights and Recommendations\n"
report += "- **Model Choice**: Linear Regression is simple and fast but may not capture nonlinearities or complex patterns in data.\n"
report += "- **Residual Analysis**:\n"
report += "  - Ideally, residuals should be randomly distributed around zero with no clear patterns.\n"
report += "  - Patterns or trends in residual plots suggest missing features or nonlinear relationships.\n"
report += "- **Regularization**: Consider Ridge or Lasso regression to reduce overfitting and improve generalization.\n"
report += "- **Feature Engineering**:\n"
report += "  - Polynomial features (degree 2 or 3) can capture nonlinear effects.\n"
report += "  - Feature selection or dimensionality reduction can remove irrelevant/noisy features.\n"
report += "- **Cross-validation**: Using k-fold (e.g., 5 or 10 folds) ensures robust performance estimates.\n"
report += "- **Hyperparameter Tuning**: For Ridge/Lasso, tune regularization parameters (`alpha`) using grid search.\n"
report += "- **Data Quality**: Ensure data preprocessing handles outliers, missing values, and scaling properly.\n"
report += "- **Model Alternatives**: For better accuracy, try nonlinear models like Random Forests, Gradient Boosting, or Support Vector Regression.\n"
report += "- **Client Variability**: Differences in metrics between clients may indicate data heterogeneity; consider personalized or federated approaches.\n"
report += "- **Performance vs Speed**: Balance model complexity and training time based on deployment constraints.\n\n"

report += "## Next Steps\n"
report += ("- Implement Ridge Regression with cross-validation and hyperparameter tuning.\n"
           "- Explore polynomial feature expansion and nonlinear models.\n"
           "- Perform feature importance analysis to refine input features.\n"
           "- Investigate and preprocess outliers or data quality issues.\n"
           "- Consider federated learning strategies if clients' data differ significantly.\n")

report_path = os.path.join(ANALYSIS_DIR, 'analysis_linear_regression.md')
with open(report_path, 'w') as f:
    f.write(report)
