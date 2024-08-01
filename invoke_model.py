import requests

# API endpoint URL
api_url = "https://domino.myalcon.com:443/models/66847885f1d8ec68cda71153/latest/model"

# Authentication credentials
auth_token = "Ss8h8qpSVEGfvW2HWRVS3wA54RCThjAcuLydllJfVZjaCwAZqJx4Zi7RfkrFKeWA"

# Example input data
input_data = [
    {
        "revenue_recognition_date": "2023-07-15",
        "is_holiday": False,
        "is_billing_day": True
    },
    {
        "revenue_recognition_date": "2023-07-16",
        "is_holiday": True,
        "is_billing_day": False
    }
]

# Optional: specify a model name
# Set to None if you want to use all models
model_name = "Atlantis_IrisInnovations"

# Prepare the JSON payload
payload = {
    "data": {
        "input_data": input_data,
        # "model_name": model_name
    }
}

# Make the API call
response = requests.post(
    api_url,
    auth=(auth_token, auth_token),
    json=payload
)

# Print the results
print(f"Status Code: {response.status_code}")
print("Headers:")
for key, value in response.headers.items():
    print(f"{key}: {value}")
print("\nResponse JSON:")
print(response.json())

# After receiving the response
if response.status_code == 200:
    result = response.json()

    print("\nPredictions:")
    if 'result' in result and isinstance(result['result'], list):
        for model_prediction in result['result']:
            print(f"\nModel: {model_prediction['model_name']}")
            for pred in model_prediction['predictions']:
                print(
                    f"Date: {pred['date']}, Prediction: {pred['prediction']:.2f}")
    else:
        print("No predictions found in the response.")

    # Print additional information
    print(
        f"\nModel execution time: {result.get('model_time_in_ms', 'N/A')} ms")
    print(f"Request ID: {result.get('request_id', 'N/A')}")

    if 'release' in result:
        print("\nRelease Information:")
        print(
            f"Harness Version: {result['release'].get('harness_version', 'N/A')}")
        print(
            f"Model Version: {result['release'].get('model_version', 'N/A')}")
        print(
            f"Model Version Number: {result['release'].get('model_version_number', 'N/A')}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)

# Example of how to access specific predictions
if response.status_code == 200:
    result = response.json()
    if 'result' in result and isinstance(result['result'], list):
        # Access predictions for a specific model (e.g., 'Atlantis_IrisInnovations')
        target_model = 'Atlantis_IrisInnovations'
        for model_prediction in result['result']:
            if model_prediction['model_name'] == target_model:
                print(f"\nPredictions for {target_model}:")
                for pred in model_prediction['predictions']:
                    print(
                        f"Date: {pred['date']}, Prediction: {pred['prediction']:.2f}")
                break
        else:
            print(f"No predictions found for model: {target_model}")

    else:
        print("No predictions found in the response.")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
