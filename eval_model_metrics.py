import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from autogluon.core import TabularDataset
from autogluon.tabular import TabularPredictor
import os
from datetime import datetime

def generate_report(fit_summary):
    """
    Generate a report summarizing the model's performance.

    Args:
    fit_summary (dict): A dictionary containing the model's fit summary.

    Returns:
    str: A formatted string containing the report.
    """
    report = ""
    # Extracting useful information
    best_model = fit_summary['leaderboard']['model'][0]
    train_time = fit_summary['leaderboard']['fit_time_marginal'][0]
    val_score = fit_summary['leaderboard']['score_val'][0]
    report += f"Best Model: {best_model}\n"
    report += f"Training Time: {train_time} seconds\n"
    report += f"Validation Score: {val_score}\n\n"
    report += "Model Performance:\n"
    report += fit_summary['leaderboard'].to_string(index=False)
    return report

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.

    Returns:
    float: MAPE value as a percentage.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.

    Returns:
    float: SMAPE value as a percentage.
    """
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

# Set the path to load the model
load_model_path = "AutogluonModels/regression/Atlantis_IrisInnovations_20240618_2223"

# Load evaluation data
evaluation_data = pd.read_csv("data/train_test/evaluation_data_Narnia_LensLogic_20240618_2010.csv")
evaluation_data_ag = TabularDataset(evaluation_data)

# Load the model
predictor = TabularPredictor.load(load_model_path)

# Generate and print the report
report = generate_report(predictor.fit_summary())
print(report)

# Save the report to a file
output_report_dir = '/mnt/code/reports'
os.makedirs(output_report_dir, exist_ok=True)
now = datetime.now()
current_time = now.strftime("%Y%m%d_%H%M")
series_id = 'Atlantis_IrisInnovations'
with open(os.path.join(output_report_dir, f'{series_id}_regression_fit_summary_report_{current_time}.txt'), 'w') as f:
    f.write(report)

# Make predictions
predictions = predictor.predict(evaluation_data_ag)

# Combine the actuals and predictions for analysis
evaluation_data['predicted_day_sales_usd'] = predictions
evaluation_data = evaluation_data.sort_values(by='revenue_recognition_date')

# Calculate and print various error metrics

# Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(evaluation_data['day_sales_usd'], evaluation_data['predicted_day_sales_usd'])
print(f"MAPE: {mape}%")

# Mean Absolute Error (MAE)
mae = mean_absolute_error(evaluation_data['day_sales_usd'], evaluation_data['predicted_day_sales_usd'])
print(f"MAE: {mae}")

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(evaluation_data['day_sales_usd'], evaluation_data['predicted_day_sales_usd']))
print(f"RMSE: {rmse}")

# Symmetric Mean Absolute Percentage Error (SMAPE)
smape = symmetric_mean_absolute_percentage_error(evaluation_data['day_sales_usd'], evaluation_data['predicted_day_sales_usd'])
print(f"SMAPE: {smape}%")
