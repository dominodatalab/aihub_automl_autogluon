import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from skimage.metrics import mean_squared_error

from feature_engineering import compute_features

def generate_report(fit_summary):
    """
    Generate a report summarizing the model's performance.

    Args:
    fit_summary (dict): A dictionary containing the model's fit summary from Autogluon.

    Returns:
    str: A formatted string containing the report.
    """
    best_model = fit_summary['leaderboard']['model'][0]
    train_time = fit_summary['leaderboard']['fit_time_marginal'][0]
    val_score = fit_summary['leaderboard']['score_val'][0]
    report = (f"Best Model: {best_model}\n"
              f"Training Time: {train_time} seconds\n"
              f"Validation Score: {val_score}\n\n"
              "Model Performance:\n"
              f"{fit_summary['leaderboard'].to_string(index=False)}")
    return report

def load_and_prepare_data(file_path):
    """
    Load and preprocess the data from a CSV file.

    Args:
    file_path (str): Path to the CSV file containing the data.

    Returns:
    pandas.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(file_path)
    df['revenue_recognition_date'] = pd.to_datetime(df['revenue_recognition_date'])
    df = df.sort_values(by='revenue_recognition_date')
    df = df[df['day_sales_usd'] >= 0]  # Remove negative sales
    return df

def split_data(df, evaluation_frac=0.2, columns_to_drop=None):
    """
    Split the data into training and evaluation sets.

    Args:
    df (pandas.DataFrame): Input DataFrame.
    evaluation_frac (float): Fraction of data to use for evaluation.
    columns_to_drop (list): List of column names to drop.

    Returns:
    tuple: (train_data, evaluation_data) as pandas DataFrames.
    """
    evaluation_data = df.sample(frac=evaluation_frac, random_state=0)
    train_data = df.drop(evaluation_data.index)
    if columns_to_drop:
        train_data = train_data.drop(columns=columns_to_drop)
        evaluation_data = evaluation_data.drop(columns=columns_to_drop)
    return train_data, evaluation_data

def train_model(train_data, label, series_id):
    """
    Train a model using AutoGluon.

    Args:
    train_data (pandas.DataFrame): Training data.
    label (str): Name of the target column.
    series_id (str): Identifier for the data series.

    Returns:
    tuple: (predictor, current_time) where predictor is the trained AutoGluon model
           and current_time is a string representing the current time.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    predictor = TabularPredictor(label=label, eval_metric='root_mean_squared_error',
                                 path=f'/mnt/artifacts/{series_id}/') # change this to where you want to save the model
    train_data_ag = TabularDataset(train_data)
    predictor.fit(train_data=train_data_ag, presets='best_quality',
                  num_cpus=6, feature_prune_kwargs={'force_prune': True}) # change num_cpus according to the hardware you have available
    return predictor, current_time

def save_report(report, series_id, current_time):
    """
    Save the generated report to a file.

    Args:
    report (str): The report to save.
    series_id (str): Identifier for the data series.
    current_time (str): Current time string.
    """
    output_report_dir = '/mnt/code/reports'
    os.makedirs(output_report_dir, exist_ok=True)
    with open(os.path.join(output_report_dir, f'{series_id}_regression_fit_summary_report_{current_time}.txt'), 'w') as f:
        f.write(report)

def plot_predictions(pred_df, df, sampled_date, days_to_forecast):
    """
    Plot actual vs predicted sales.

    Args:
    pred_df (pandas.DataFrame): Predicted data.
    df (pandas.DataFrame): Full dataset.
    sampled_date (str): Start date for predictions.
    days_to_forecast (int): Number of days to forecast.
    """
    plt.figure(figsize=(12, 6))
    # Plot predicted values
    plt.plot(pred_df['date'], pred_df['predicted'], label='Predicted', color='red', marker='x')
    for i, row in pred_df.iterrows():
        plt.annotate(f"{row['predicted']:.2f}", (row['date'], row['predicted']),
                     textcoords="offset points", xytext=(0, 10), ha='center')
    
    # Plot actual values
    start_date = sampled_date
    end_date = pd.to_datetime(start_date) + pd.Timedelta(days=days_to_forecast-1)
    filtered_data = df[(df['revenue_recognition_date'] >= start_date) & (df['revenue_recognition_date'] <= end_date)]
    plt.plot(filtered_data['revenue_recognition_date'], filtered_data['day_sales_usd'], label='Actual', marker='o', color='blue')
    for i, row in filtered_data.iterrows():
        plt.annotate(f"{row['day_sales_usd']:.2f}", (row['revenue_recognition_date'], row['day_sales_usd']),
                     textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.xlabel('Date')
    plt.ylabel('Day Sales USD')
    plt.title('Actual vs Predicted Day Sales USD')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_predictions(predictor, train_df, sampled_date, forecast_periods=14):
    """
    Generate predictions for future dates.

    Args:
    predictor: Trained AutoGluon predictor.
    train_df (pandas.DataFrame): Training data.
    sampled_date (str): Start date for predictions.
    forecast_periods (int): Number of periods to forecast.

    Returns:
    pandas.DataFrame: DataFrame containing the predictions.
    """
    forecast_dates = pd.date_range(start=sampled_date, periods=forecast_periods+1).tolist()
    predictions = []
    for date in forecast_dates:
        new_row = {'revenue_recognition_date': date, 'is_holiday': False, 'is_billing_day': True}
        temp_df = pd.concat([train_df, pd.DataFrame([new_row])], ignore_index=True)
        temp_df['revenue_recognition_date'] = pd.to_datetime(temp_df['revenue_recognition_date'])
        temp_df = compute_features(temp_df)
        features = temp_df[temp_df['revenue_recognition_date'] == date]
        features = features.drop(columns=['day_sales_usd'])
        prediction = predictor.predict(features)
        predictions.append((date, prediction.values[0]))
        new_row.update(features.iloc[0].to_dict())
        new_row['day_sales_usd'] = prediction.values[0]
        train_df = pd.concat([train_df, pd.DataFrame([new_row])], ignore_index=True)
    pred_df = pd.DataFrame(predictions, columns=['date', 'predicted'])
    return pred_df

def load_model(load_model_path):
    """
    Load a previously trained model.

    Args:
    load_model_path (str): Path to the saved model.

    Returns:
    TabularPredictor: Loaded AutoGluon predictor.
    """
    return TabularPredictor.load(load_model_path)

def evaluate_model(predictor, evaluation_data_ag):
    """
    Evaluate the model's performance.

    Args:
    predictor: Trained AutoGluon predictor.
    evaluation_data_ag (TabularDataset): Evaluation data as a TabularDataset.
    """
    predictions = predictor.predict(evaluation_data_ag)
    evaluation_data['predicted_day_sales_usd'] = predictions
    evaluation_data = evaluation_data.sort_values(by='revenue_recognition_date')
    rmse = np.sqrt(mean_squared_error(evaluation_data['day_sales_usd'], evaluation_data['predicted_day_sales_usd']))
    print(f"RMSE: {rmse}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(evaluation_data['revenue_recognition_date'], evaluation_data['day_sales_usd'], label='Actual', marker='o')
    plt.plot(evaluation_data['revenue_recognition_date'], evaluation_data['predicted_day_sales_usd'], label='Predicted', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Day Sales USD')
    plt.title('Actual vs Predicted Day Sales USD')
    plt.legend()
    plt.grid(True)
    plt.show()

def model_analysis(predictor, train_df_path, eval_df_path, df, forecast_periods=14):
    """
    Analyze the model's performance by plotting actuals vs forecast.

    Args:
    predictor: Trained AutoGluon predictor.
    train_df_path (str): Path to the training data CSV.
    eval_df_path (str): Path to the evaluation data CSV.
    df (pandas.DataFrame): Full dataset.
    forecast_periods (int): Number of periods to forecast.
    """
    train_df = pd.read_csv(train_df_path)
    evaluation_data = pd.read_csv(eval_df_path)
    random_index = np.random.randint(0, len(evaluation_data))
    sampled_date = evaluation_data['revenue_recognition_date'].iloc[random_index]
    pred_df = get_predictions(predictor, train_df, sampled_date, forecast_periods)
    plot_predictions(pred_df, df, sampled_date, forecast_periods)

def main():
    """
    Main function to orchestrate the entire process of loading data,
    training/loading a model, and analyzing its performance.
    """
    series_id = 'Narnia_LensLogic'
    # file_path = f'/mnt/code/data/feature_engineering/{series_id}_processed_features.csv' # read from a dataset
    file_path = f'data/feature_engineering/{series_id}_processed_features.csv' # use data in local folder
    df = load_and_prepare_data(file_path)
    label = 'day_sales_usd'
    load_model_path = None
    columns_to_drop = ['quarterly_cos', 'week_of_month', 'quarterly_sin', 'quarter', 'part_of_month',
                       'first_week', 'rolling_mean_same_day_of_week_84_prev_month', 'latest_same_day_of_week_intramonth']

    if not load_model_path:
        # Train a new model
        train_data, evaluation_data = split_data(df, columns_to_drop=columns_to_drop)
        
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
        # Make sure the train_test folder exists at the location where the file is going to be saved
        # train_file_name = f"/mnt/code/data/train_test/train_data_{series_id}_{current_datetime}.csv" # dataset location to save the training data
        train_file_name = f"data/train_test/train_data_{series_id}_{current_datetime}.csv" # save training data to local file
        # evaluation_file_name = f"/mnt/code/data/train_test/evaluation_data_{series_id}_{current_datetime}.csv" #dataset location to save the evaluation data
        evaluation_file_name = f"data/train_test/evaluation_data_{series_id}_{current_datetime}.csv" # save evaluation data to local file
        
        # Save the training and evaluation data
        train_data.to_csv(train_file_name, index=False)
        evaluation_data.to_csv(evaluation_file_name, index=False)

        predictor, current_time = train_model(train_data, label, series_id)
        report = generate_report(predictor.fit_summary())
        print(report)
        save_report(report, series_id, current_time)
    else:
        # Load an existing model and analyze its performance
        predictor = load_model(load_model_path)
        model_analysis(predictor, "data/train_test/train_data_Atlantis_OptiGlimpse_20240618_2119.csv", #change these paths and file names to where your train_test dataset resides
                       "data/train_test/evaluation_data_Atlantis_OptiGlimpse_20240618_2119.csv", df, forecast_periods=14)

if __name__ == "__main__":
    main()
