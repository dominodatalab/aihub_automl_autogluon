import argparse
import glob
import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from autogluon.tabular import TabularPredictor
from feature_engineering import compute_features

def load_model(load_model_path):
    """
    Load an AutoGluon TabularPredictor model from the specified path.

    Args:
        load_model_path (str): Path to the saved model.

    Returns:
        TabularPredictor: Loaded AutoGluon model.
    """
    return TabularPredictor.load(load_model_path, require_version_match=False, require_py_version_match=False)

def read_csv_input(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a CSV file and convert it to a list of dictionaries.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        List[Dict[str, Any]]: List of dictionaries representing the CSV data.
    """
    df = pd.read_csv(file_path)
    return df.to_dict('records')

def write_predictions_to_file(predictions: List[Dict[str, Any]], output_dir: str, series_id: Optional[str] = None):
    """
    Write prediction results to a JSON file.

    Args:
        predictions (List[Dict[str, Any]]): List of prediction results.
        output_dir (str): Directory to write the output file.
        series_id (Optional[str]): Identifier for the series, used in the filename.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if series_id:
        filename = f"{series_id}_{timestamp}.json"
    else:
        filename = f"all_series_{timestamp}.json"

    # Write the predictions to the file
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"Predictions written to {file_path}")

# Define the paths and names for the 5 models
MODELS = {
    "Atlantis_IrisInnovations": "/mnt/artifacts/Atlantis_IrisInnovations",
    "Atlantis_OptiGlimpse": "/mnt/artifacts/Atlantis_OptiGlimpse",
    "Narnia_LensLogic": "/mnt/artifacts/Narnia_LensLogic",
    "Narnia_OptiGlimpse": "/mnt/artifacts/Narnia_OptiGlimpse",
    "Rivendell_SightSphere": "/mnt/artifacts/Rivendell_SightSphere"
}

# Load all models and their corresponding training data
predictors = {}
train_dfs = {}

for name, path in MODELS.items():
    predictors[name] = load_model(path)
    # Load the unsplit training data file
    train_df_path = glob.glob(f"/mnt/code/data/feature_engineering/{name}_processed_features.csv")[0]
    train_dfs[name] = pd.read_csv(train_df_path)
    train_dfs[name]['revenue_recognition_date'] = pd.to_datetime(train_dfs[name]['revenue_recognition_date'])
    train_dfs[name] = train_dfs[name].sort_values('revenue_recognition_date').reset_index(drop=True)

def predict_with_models(input_data: Union[Dict[str, Any], List[Dict[str, Any]]], model_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Make predictions using specified model or all loaded models, with dictionary input.

    Args:
        input_data (Union[Dict[str, Any], List[Dict[str, Any]]]): Input data as a dictionary or a list of dictionaries.
        model_name (Optional[str]): Name of the specific model to use. If None, use all models.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing predictions from each model, including dates.
    """
    try:
        # Ensure input_data is a list of dictionaries
        if isinstance(input_data, dict):
            input_data = [input_data]
        elif not isinstance(input_data, list):
            raise ValueError("Invalid input. Expected a dictionary or a list of dictionaries.")

        results = []
        models_to_use = {model_name: predictors[model_name]} if model_name else predictors

        if model_name and model_name not in predictors:
            return [{"error": f"Model '{model_name}' not found"}]

        for name, predictor in models_to_use.items():
            try:
                predictions = []
                for row in input_data:
                    # Extract required fields from input
                    date = pd.to_datetime(row['revenue_recognition_date'])
                    is_holiday = row.get('is_holiday', False)
                    is_billing_day = row.get('is_billing_day', True)

                    # Check if the date already exists in the training dataframe
                    existing_data = train_dfs[name][train_dfs[name]['revenue_recognition_date'] == date]

                    if not existing_data.empty:
                        # If the date exists, use the existing value
                        prediction = existing_data['day_sales_usd'].tolist()[0]
                    else:
                        # If the date doesn't exist, compute features and make a prediction
                        new_row = {'revenue_recognition_date': date,
                                   'is_holiday': is_holiday,
                                   'is_billing_day': is_billing_day}
                        temp_df = pd.concat([train_dfs[name], pd.DataFrame([new_row])], ignore_index=True)

                        # Sort the temporary DataFrame
                        temp_df = temp_df.sort_values('revenue_recognition_date').reset_index(drop=True)

                        # Compute features
                        temp_df = compute_features(temp_df)

                        # Filter features for the specific date
                        features = temp_df[temp_df['revenue_recognition_date'] == date]
                        features = features.drop(columns=['day_sales_usd'])

                        # Make prediction using AutoGluon's TabularPredictor
                        prediction = predictor.predict(features).tolist()[0]

                    predictions.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "prediction": prediction
                    })

                results.append({
                    "model_name": name,
                    "predictions": predictions
                })
            except Exception as e:
                results.append({
                    "model_name": name,
                    "error": str(e)
                })

        return results

    except Exception as e:
        return [{"error": str(e)}]

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using the multi-model predictor.")
    parser.add_argument("--is_job", action="store_true", help="Specify if this is a job run")
    parser.add_argument("--series_id", type=str, help="Series ID to use as the model name (optional)")
    parser.add_argument("--path", type=str, help="Path to the input CSV file")
    parser.add_argument("--output_dir", type=str, default="/mnt/data/forecasts/",
                        help="Directory to write output files (default: /mnt/data/forecasts/)")
    args = parser.parse_args()

    if args.is_job:
        if not args.path:
            print("Error: --path must be specified when --is_job is used.")
            exit(1)

        # Read the CSV file
        input_data = read_csv_input(args.path)

        # Make predictions
        predictions = predict_with_models(input_data, model_name=args.series_id)

        # Write predictions to file
        write_predictions_to_file(predictions, args.output_dir, args.series_id)

    else:
        # Interactive example
        print("Running interactive example...")

        # Single row example
        single_row_input = {
            "revenue_recognition_date": "2023-06-24",
            "is_holiday": False,
            "is_billing_day": True
        }
        print("Predicting with single row:")
        single_predictions = predict_with_models(single_row_input)
        write_predictions_to_file(single_predictions, args.output_dir, "interactive_single")

        # Multiple rows example
        multiple_rows_input = [
            {"revenue_recognition_date": "2023-06-24", "is_holiday": False, "is_billing_day": True},
            {"revenue_recognition_date": "2023-06-25", "is_holiday": True, "is_billing_day": False},
            {"revenue_recognition_date": "2023-06-26", "is_holiday": False, "is_billing_day": True}
        ]
        print("\nPredicting with multiple rows:")
        multiple_predictions = predict_with_models(multiple_rows_input)
        write_predictions_to_file(multiple_predictions, args.output_dir, "interactive_multiple")

        # Specific model example
        print("\nPredicting with a specific model (Atlantis_IrisInnovations):")
        specific_predictions = predict_with_models(single_row_input, model_name="Atlantis_IrisInnovations")
        write_predictions_to_file(specific_predictions, args.output_dir, "interactive_Atlantis_IrisInnovations")

    print(f"All predictions have been written to files in {args.output_dir}")
