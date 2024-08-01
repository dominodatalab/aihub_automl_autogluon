import os
import pandas as pd
import numpy as np
from autogluon.common import space
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe
import random
from datetime import datetime
from matplotlib import pyplot as plt

# Set a random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)


# Generate a report from fit_summary
def generate_report(fit_summary):
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


# Load the dataset
df = pd.read_csv('data/obfuscated_6_series_2021-2024_billing_and_holidays.csv')
# remove market_l5,product_l3_desc,day_name from df
# df = df.drop(columns=['market_l5', 'product_l3_desc', 'day_name'])

# Process each unique series_id
for unique_series_id in df['series_id'].unique():

    # if unique_series_id is not Narnia_LensLogic skip
    if unique_series_id != 'Narnia_LensLogic':
        continue
    df_subset = df[df['series_id'] == unique_series_id].copy()

    # Drop rows where day_sales_usd is negative
    df_subset = df_subset[df_subset['day_sales_usd'] >= 0]

    # Preprocess the data
    df_subset['revenue_recognition_date'] = pd.to_datetime(df_subset['revenue_recognition_date'], errors='coerce')
    if not np.issubdtype(df_subset['revenue_recognition_date'].dtype, np.datetime64):
        raise ValueError("revenue_recognition_date column could not be converted to datetime")

    df_subset['day_of_week'] = df_subset['revenue_recognition_date'].dt.dayofweek
    df_subset = df_subset.rename(columns={
        'series_id': 'item_id',
        'revenue_recognition_date': 'timestamp',
        'day_sales_usd': 'target'
    })
    df_subset['day_of_week'] = df_subset['day_of_week'].astype('category')
    df_subset['is_holiday'] = df_subset['is_holiday'].astype('category')
    df_subset['is_billing_day'] = df_subset['is_billing_day'].astype('category')

    # Split the data into training and test sets
    train_end_date = df_subset['timestamp'].max() - pd.Timedelta(days=7)
    train_data = df_subset[df_subset['timestamp'] <= train_end_date]
    test_data = df_subset[df_subset['timestamp'] > train_end_date]

    # Convert to TimeSeriesDataFrame
    train_ts_data = TimeSeriesDataFrame.from_data_frame(train_data, id_column='item_id', timestamp_column='timestamp')
    test_ts_data = TimeSeriesDataFrame.from_data_frame(test_data, id_column='item_id', timestamp_column='timestamp')
    train_ts_data = train_ts_data.convert_frequency(freq="D").fill_missing_values()

    # Set the current time for filenames
    now = datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")

    # Choose whether to train a new model or load an existing one
    # load_model_path = "AutogluonModels/Atlantis_OptiGlimpse_20240611_124001"  # Set this to the path if you want to load an existing model, else set to None
    load_model_path = None

    if load_model_path:
        predictor = TimeSeriesPredictor.load(load_model_path)
    else:
        predictor = TimeSeriesPredictor(
            target='target',
            prediction_length=7,
            freq='D',
            eval_metric="WAPE",
            path=f'AutogluonModels/{unique_series_id}_{current_time}',
            known_covariates_names=['is_holiday', 'is_billing_day'],
        )
        predictor.fit(train_data=train_ts_data, presets='high_quality', time_limit=60 * 60,
                      hyperparameters={
                          "DeepAR": {},
                          "ETS": {},
                          "AutoARIMA": {},
                          "TemporalFusionTransformer": {},
                          # eg of running non default HPO below
                          # "TemporalFusionTransformer": [
                          #     {"context_length": space.Int(8, 64)},
                          #     {"num_heads": space.Int(4, 8)},
                          # ],
                      },
                      # hyperparameter_tune_kwargs={
                      #     "num_trials": 20,
                      #     "scheduler": "local",
                      #     "searcher": "random",
                      # },
                      enable_ensemble=True,
                      )

    # Generate and print the report
    report = generate_report(predictor.fit_summary())
    print(report)

    # Save the report to a file
    output_report_dir = 'reports/ts_models'
    os.makedirs(output_report_dir, exist_ok=True)
    if not load_model_path:
        with open(os.path.join(output_report_dir, f'{unique_series_id}_fit_summary_report_{current_time}.txt'),
                  'w') as f:
            f.write(report)

    # Forecasting
    future_index = get_forecast_horizon_index_ts_dataframe(train_ts_data, prediction_length=14, freq='D')

    known_covariates = pd.DataFrame(index=future_index)
    # known_covariates["is_holiday"] = [False] * 14
    # known_covariates["is_billing_day"] = [True] * 14

    # set of known covariates for Atlantis_OptiGlimpse
    # known_covariates["is_holiday"] = [False] * 14
    # known_covariates["is_billing_day"] = [True, True, True, True, True, True, False, True, True, True, True, True,
    # True, True]

    # set of known covariates for Narnia_LensLogic
    known_covariates["is_holiday"] = [False] * 14
    known_covariates["is_billing_day"] = [True, True, True, True, False, False, False, True, True, True, True, True, True, True]

    known_covariates = TimeSeriesDataFrame.from_data_frame(known_covariates.astype('category'))

    # Evaluate the model on the test set
    test_predictions = predictor.predict(train_ts_data, known_covariates=known_covariates)
    print("Test Predictions:")
    print(test_predictions)

    # Combine train, test, and predictions into respective series
    train_series = pd.Series(train_data['target'].values, index=train_data['timestamp']).sort_index()[-30:]
    test_series = pd.Series(test_data['target'].values, index=test_data['timestamp']).sort_index()
    pred_series = pd.Series(test_predictions['mean'].values,
                            index=test_predictions.index.get_level_values('timestamp')).sort_index()

    # Ensure we only plot the matching timestamps for test and prediction series
    common_index = test_series.index.intersection(pred_series.index)
    test_series = test_series.loc[common_index]
    pred_series = pred_series.loc[common_index]

    # Extract the prediction intervals for the common indices
    pred_lower = test_predictions.loc[
        test_predictions.index.get_level_values('timestamp').isin(common_index), '0.1'].sort_index()
    pred_upper = test_predictions.loc[
        test_predictions.index.get_level_values('timestamp').isin(common_index), '0.9'].sort_index()

    # Combine the last point of the training series with the test series for a continuous line
    combined_series = pd.concat([train_series, test_series])

    # Plot the last 30 days of training data, combined series, and predictions
    plt.figure(figsize=(12, 6))
    plt.plot(combined_series.index, combined_series.values, linestyle='-', label='Actual - Training + Test')
    plt.plot(test_series.index, test_series.values, marker='o', linestyle='None', label='Actual - Test')
    plt.plot(pred_series.index, pred_series.values, marker='x', linestyle='-', label='Predicted')
    plt.fill_between(pred_series.index,
                     pred_lower.values,
                     pred_upper.values,
                     color='gray', alpha=0.2, label='Prediction Interval')
    plt.xlabel('Date')
    plt.ylabel('Sales (USD)')
    plt.title('Actual vs Predicted Sales')
    plt.legend()

    # Save the plot to a file
    output_plot_dir = 'plots/ts_models/'
    os.makedirs(output_plot_dir, exist_ok=True)
    output_file = os.path.join(output_plot_dir, f'{unique_series_id}_{current_time}.png')
    plt.savefig(output_file)
    plt.close()

    plt.show()
