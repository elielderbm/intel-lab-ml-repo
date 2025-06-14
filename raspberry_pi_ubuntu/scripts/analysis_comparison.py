import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns

# Configuração dos diretórios
BASE_DIR = os.path.dirname(__file__)
MODELS = {
    'CNN': os.path.join(BASE_DIR, '../ml_results_cnn'),
    'Random Forest': os.path.join(BASE_DIR, '../ml_results_random_forest'),
    'Linear Regression': os.path.join(BASE_DIR, '../ml_results_linear_regression')
}
ANALYSIS_DIR = os.path.join(BASE_DIR, '../ml_results_comparison')

# Criar diretório de análise com verificação
try:
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
except Exception as e:
    print(f"Error creating directory {ANALYSIS_DIR}: {e}")
    raise

# Carregar resultados
all_results = {model: {} for model in MODELS}
total_times = {model: {} for model in MODELS}

for model, output_dir in MODELS.items():
    results_files = glob.glob(os.path.join(output_dir, 'results_client*.json'))
    time_files = glob.glob(os.path.join(output_dir, 'total_time_client*.txt'))
    
    for file in results_files:
        client_id = os.path.basename(file).split('_')[1].split('.')[0]
        with open(file, 'r') as f:
            all_results[model][client_id] = json.load(f)
    
    for file in time_files:
        client_id = os.path.basename(file).split('_')[2].split('.')[0]
        with open(file, 'r') as f:
            total_times[model][client_id] = float(f.read().split(': ')[1].split(' ')[0])

# Verificar consistência dos clientes
clients = set()
for model in MODELS:
    clients.update(all_results[model].keys())
clients = sorted(list(clients))

# Métricas a comparar
metrics = ['rmse', 'mae', 'r2', 'training_time']
metrics_labels = {
    'rmse': 'Root Mean Squared Error (RMSE)',
    'mae': 'Mean Absolute Error (MAE)',
    'r2': 'R² Score',
    'training_time': 'Training Time (seconds)',
    'total_time': 'Total Execution Time (seconds)'
}

# Gráficos comparativos por métrica (barras)
for metric in metrics + ['total_time']:
    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    x = np.arange(len(clients))
    
    for i, model in enumerate(MODELS):
        if metric == 'total_time':
            values = [total_times[model].get(c, np.nan) for c in clients]
        else:
            values = [all_results[model].get(c, {}).get(metric, np.nan) for c in clients]
        plt.bar(x + i * bar_width, values, bar_width, label=model, edgecolor='black')
    
    plt.xlabel('Client')
    plt.ylabel(metrics_labels[metric])
    plt.title(f'{metrics_labels[metric]} Across Clients')
    plt.xticks(x + bar_width, clients)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, f'comparison_{metric}.png'))
    plt.close()

# Box plots para distribuição de métricas
for metric in metrics + ['total_time']:
    plt.figure(figsize=(10, 6))
    data = []
    for model in MODELS:
        if metric == 'total_time':
            values = [total_times[model].get(c, np.nan) for c in clients]
        else:
            values = [all_results[model].get(c, {}).get(metric, np.nan) for c in clients]
        data.append(values)
    plt.boxplot(data, tick_labels=list(MODELS.keys()), patch_artist=True)
    plt.ylabel(metrics_labels[metric])
    plt.title(f'Distribution of {metrics_labels[metric]} Across Models')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, f'boxplot_{metric}.png'))
    plt.close()

# Radar chart para comparação holística
def radar_chart():
    categories = metrics + ['total_time']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    for model in MODELS:
        values = []
        for metric in categories:
            if metric == 'total_time':
                vals = [total_times[model].get(c, np.nan) for c in clients]
            else:
                vals = [all_results[model].get(c, {}).get(metric, np.nan) for c in clients]
            values.append(np.nanmean(vals))
        # Normalizar valores para radar chart
        values = [(val - min(values)) / (max(values) - min(values) + 1e-6) for val in values]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metrics_labels[m] for m in categories])
    plt.title('Model Comparison (Normalized Metrics)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'radar_comparison.png'))
    plt.close()

radar_chart()

# Análise de resíduos por cliente
for client in clients:
    plt.figure(figsize=(12, 6))
    for model in MODELS:
        if client in all_results[model]:
            results = all_results[model][client]
            if 'y_test' in results and 'y_pred' in results:
                residuals = np.array(results['y_test']) - np.array(results['y_pred'])
                sns.kdeplot(residuals, label=model, linewidth=2)
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Density')
    plt.title(f'Residual Distribution for Client {client}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, f'residuals_client_{client}.png'))
    plt.close()

# Análise estatística (teste t pareado)
stat_results = []
model_list = list(MODELS.keys())  # Convert dictionary keys to list for slicing
for metric in metrics:
    for i, model1 in enumerate(model_list[:-1]):
        for model2 in model_list[i+1:]:
            vals1 = [all_results[model1].get(c, {}).get(metric, np.nan) for c in clients]
            vals2 = [all_results[model2].get(c, {}).get(metric, np.nan) for c in clients]
            # Remover NaNs
            valid_pairs = [(v1, v2) for v1, v2 in zip(vals1, vals2) if not np.isnan(v1) and not np.isnan(v2)]
            if len(valid_pairs) > 1:
                v1, v2 = zip(*valid_pairs)
                stat, p_value = ttest_rel(v1, v2)
                stat_results.append({
                    'Metric': metric,
                    'Comparison': f'{model1} vs {model2}',
                    'Statistic': stat,
                    'p-value': p_value,
                    'Significant': p_value < 0.05
                })

df_stats = pd.DataFrame(stat_results)
stats_path = os.path.join(ANALYSIS_DIR, 'statistical_tests.csv')
try:
    df_stats.to_csv(stats_path, index=False)
except Exception as e:
    print(f"Error saving statistical_tests.csv: {e}")
    raise

# Tabela comparativa
table_data = []
for client in clients:
    for model in MODELS:
        row = {'Client': client, 'Model': model}
        for metric in metrics:
            row[metric.upper()] = all_results[model].get(client, {}).get(metric, np.nan)
        row['TOTAL_TIME'] = total_times[model].get(client, np.nan)
        table_data.append(row)

df = pd.DataFrame(table_data)
table_path = os.path.join(ANALYSIS_DIR, 'metrics_comparison.csv')
try:
    df.to_csv(table_path, index=False)
except Exception as e:
    print(f"Error saving metrics_comparison.csv: {e}")
    raise

# Gráfico de dispersão: RMSE vs Tempo de Treinamento
plt.figure(figsize=(10, 6))
for model in MODELS:
    rmse = [all_results[model].get(c, {}).get('rmse', np.nan) for c in clients]
    train_time = [all_results[model].get(c, {}).get('training_time', np.nan) for c in clients]
    plt.scatter(train_time, rmse, label=model, s=100, alpha=0.7)
plt.xlabel('Training Time (seconds)')
plt.ylabel('RMSE')
plt.title('RMSE vs Training Time Across Models')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'rmse_vs_training_time.png'))
plt.close()

# Importância das features (apenas para Random Forest)
if 'Random Forest' in MODELS:
    plt.figure(figsize=(10, 6))
    for client in clients:
        if client in all_results['Random Forest']:
            results = all_results['Random Forest'][client]
            if 'feature_importances' in results and 'feature_names' in results:
                importances = np.array(results['feature_importances'])
                feature_names = results['feature_names']
                indices = np.argsort(importances)[::-1]
                plt.bar([f'{client}_{name}' for name in np.array(feature_names)[indices]], 
                        importances[indices], alpha=0.5, label=f'Client {client}')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance Across Clients - Random Forest')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'random_forest_feature_importance.png'))
    plt.close()

# Relatório em Markdown
report = "# Machine Learning Model Comparison Report\n\n"

report += "## Introduction for Everyone\n"
report += "This report compares three machine learning models—**Linear Regression**, **Random Forest**, and **Convolutional Neural Network (CNN)**—used to predict something (like tomorrow’s temperature) for different clients (think of clients as different locations or devices). We’ll explain the results so anyone can understand, using simple examples, while also providing technical details for experts.\n\n"
report += "Imagine three chefs cooking the same dish:\n"
report += "- **Linear Regression** is like a chef following a simple recipe with one main ingredient. It’s quick but might not capture complex flavors.\n"
report += "- **Random Forest** is like a team of chefs voting on the best recipe. It combines many ideas to make a tasty dish, good for tricky flavors.\n"
report += "- **CNN** is like a master chef analyzing every detail of the ingredients. It’s great for complex dishes but takes longer to cook.\n\n"
report += "We judge these chefs on **accuracy** (how close their predictions are to the real value) and **speed** (how fast they work).\n\n"

report += "## What We Measured (Explained Simply)\n"
report += "We looked at five key numbers to compare the models:\n"
report += "- **RMSE (Root Mean Squared Error)**: How far off the predictions are, on average. Lower is better. Think of it as how much a chef’s dish misses the perfect taste (e.g., predicting 25°C when it’s 27°C).\n"
report += "- **MAE (Mean Absolute Error)**: Similar to RMSE, but simpler—it’s the average mistake size. Lower is better.\n"
report += "- **R² Score**: How well the model explains the data. Closer to 1 is better. It’s like saying, “This chef’s dish is 90% perfect!”\n"
report += "- **Training Time**: How long it takes to prepare the model (like a chef practicing the recipe). Shorter is better.\n"
report += "- **Total Execution Time**: The total time for everything (practicing and cooking). Shorter is better.\n\n"

report += "## Summary of Results (For Everyone)\n"
report += "Here’s how the models performed, averaged across all clients:\n"
report += df.groupby('Model').mean(numeric_only=True).round(4).to_markdown() + "\n\n"
report += "- **Linear Regression**: Fastest chef, but sometimes misses the mark (higher errors). Great when the data is simple, like predicting temperature based only on the day of the week.\n"
report += "- **Random Forest**: Middle ground—pretty accurate and not too slow. It’s like a reliable chef who balances taste and time.\n"
report += "- **CNN**: Most accurate chef for complex data, but takes the longest. Perfect for tricky predictions, like using weather patterns or images.\n\n"
report += "Think of choosing a car: Linear Regression is a bicycle (fast, simple), Random Forest is a sedan (reliable, versatile), and CNN is a sports car (powerful, resource-heavy).\n\n"

report += "## Technical Performance Metrics\n"
report += "For experts, here’s the detailed average performance across clients:\n"
report += df.groupby('Model').mean(numeric_only=True).round(4).to_markdown() + "\n"
report += "- **RMSE and MAE**: Lower values indicate better prediction accuracy. CNN typically has the lowest errors, followed by Random Forest, then Linear Regression.\n"
report += "- **R²**: Values closer to 1 show better model fit. CNN and Random Forest often outperform Linear Regression.\n"
report += "- **Training and Total Time**: Linear Regression is fastest, followed by Random Forest. CNN requires significantly more time due to its complexity.\n\n"

report += "## Statistical Analysis\n"
report += "We used paired t-tests to check if differences between models are significant:\n"
report += df_stats.round(4).to_markdown() + "\n"
report += "- **Explanation**: A p-value < 0.05 means the difference is likely real, not random. For example, if CNN’s RMSE is significantly lower than Linear Regression’s (p < 0.05), CNN is reliably more accurate.\n"
report += "- **For Everyone**: This is like tasting two dishes and being 95% sure one is better. The table shows where one model clearly beats another.\n\n"

report += "## Visual Analysis\n"
report += "We created several plots to understand the results:\n"
report += "- **Bar Plots** (`comparison_<metric>.png`): Show how each model performs for each client. Like comparing chefs’ scores across different kitchens.\n"
report += "- **Box Plots** (`boxplot_<metric>.png`): Show the range of performance. A tight box means consistent results; a wide box means varied results.\n"
report += "- **Radar Chart** (`radar_comparison.png`): A “star” shape comparing all metrics at once. Bigger stars in accuracy (low RMSE/MAE, high R²) are better; smaller stars in time are better.\n"
report += "- **Residual Plots** (`residuals_client_<client>.png`): Show prediction errors. Ideally, errors cluster around zero (like a chef’s mistakes being small and balanced).\n"
report += "- **RMSE vs Training Time** (`rmse_vs_training_time.png`): Shows trade-offs. Points lower and left are better (accurate and fast).\n"
report += "- **Feature Importance (Random Forest)** (`random_forest_feature_importance.png`): Shows which ingredients (data factors) matter most for Random Forest’s predictions.\n\n"

report += "## Insights (Simple and Technical)\n"
report += "- **CNN**:\n"
report += "  - **Simple**: Like a master chef, it makes the best dish for complex recipes (e.g., predicting from images or patterns) but needs more time and tools.\n"
report += "  - **Technical**: Excels in high-dimensional data (e.g., time series, images). Lowest RMSE/MAE but highest training time. Check convergence plots for overfitting.\n"
report += "- **Random Forest**:\n"
report += "  - **Simple**: Like a team of chefs agreeing on a recipe, it’s reliable and good for most situations, balancing taste and time.\n"
report += "  - **Technical**: Handles nonlinear data well, provides feature importance. Moderate RMSE/MAE and time. Residual asymmetry may indicate bias.\n"
report += "- **Linear Regression**:\n"
report += "  - **Simple**: Like a quick chef with a basic recipe, it’s fast but may not nail complex flavors.\n"
report += "  - **Technical**: Fastest model, but limited by linearity. Higher RMSE/MAE; residual patterns suggest nonlinear features needed.\n"
report += "- **Client Differences**:\n"
report += "  - **Simple**: Each client (location/device) has different data, like cooking in different kitchens. Some models work better in certain kitchens.\n"
report += "  - **Technical**: Significant metric differences (p < 0.05) suggest data heterogeneity. Federated learning (FedAvg, FedProx) could help.\n"
report += "- **Trade-offs**:\n"
report += "  - **Simple**: Choose Linear Regression for speed, Random Forest for balance, or CNN for accuracy.\n"
report += "  - **Technical**: CNN offers high accuracy at computational cost; Linear Regression is fastest but less accurate; Random Forest balances both.\n\n"

report += "## Recommendations\n"
report += "- **For Everyone**:\n"
report += "  - **Clean Data**: Make sure the ingredients (data) are fresh and consistent, like washing vegetables before cooking.\n"
report += "  - **Choose Wisely**: Pick the chef (model) based on your needs—speed (Linear Regression), balance (Random Forest), or precision (CNN).\n"
report += "  - **Test Thoroughly**: Try the dish in new kitchens (new data) to ensure it’s good everywhere.\n"
report += "- **Technical**:\n"
report += "  - **Data Preprocessing**: Standardize scaling, handle outliers/missing values.\n"
report += "  - **Hyperparameter Tuning**: CNN (learning rate, layers), Random Forest (`n_estimators`, `max_depth`), Linear Regression (Ridge/Lasso `alpha`).\n"
report += "  - **Feature Engineering**: Polynomial features for Linear Regression; use Random Forest feature importance for selection.\n"
report += "  - **Federated Learning**: Use FedAvg/FedProx for client data variability.\n"
report += "  - **Deployment**: Linear Regression for low-resource settings; CNN for high-accuracy needs.\n\n"

report += "## Next Steps\n"
report += "- **For Everyone**: Try tweaking the recipes (model settings) and test on new data to improve results.\n"
report += "- **Technical**: Optimize hyperparameters (GridSearchCV), explore ensembles (stacking CNN/Random Forest), investigate data quality, test model compression (knowledge distillation), and validate with cross-validation.\n\n"

report += "## Generated Files\n"
report += "Check these files for more details:\n"
for metric in metrics + ['total_time']:
    report += f"- `comparison_{metric}.png` (bar plots)\n"
    report += f"- `boxplot_{metric}.png` (box plots)\n"
report += "- `radar_comparison.png` (holistic comparison)\n"
for client in clients:
    report += f"- `residuals_client_{client}.png` (error distribution)\n"
report += "- `rmse_vs_training_time.png` (accuracy vs speed)\n"
report += "- `random_forest_feature_importance.png` (key factors for Random Forest)\n"
report += "- `metrics_comparison.csv` (detailed metrics)\n"
report += "- `statistical_tests.csv` (statistical comparisons)\n"

# Salvar relatório com verificação de erros
report_path = os.path.join(ANALYSIS_DIR, 'analysis_ml_comparison.md')
try:
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Markdown report successfully saved to {report_path}")
except Exception as e:
    print(f"Error saving Markdown report to {report_path}: {e}")
    raise

print("Enhanced comparative analysis with beginner-friendly explanations completed successfully.")