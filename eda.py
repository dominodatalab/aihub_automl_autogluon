import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


def load_data(file_path, n_days):
    """
    Load the data from a CSV file and return the last n days of data.
    """
    data = pd.read_csv(file_path)
    # print number of rows
    last_n_days_data = data.tail(n_days)
    original_row_count = last_n_days_data.shape[0]
    last_n_days_data = last_n_days_data[last_n_days_data['day_sales_usd'] >= 0]
    print(f"{original_row_count - last_n_days_data.shape[0]} rows dropped")
    return last_n_days_data


def plot_time_series(data, title):
    """
    Plot the time series data.
    """
    plt.plot(data["revenue_recognition_date"], data['day_sales_usd'])
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def is_stationary(sales_data):
    """
    Perform the Augmented Dickey-Fuller test to check if the data is stationary.
    """
    adf_result = adfuller(sales_data)
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]

    print(f'ADF Statistic: {adf_statistic}')
    print(f'p-value: {p_value}')
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value}')

    if p_value < 0.05:
        print("The data is stationary (reject the null hypothesis).")
        return True
    else:
        print("The data is not stationary (fail to reject the null hypothesis).")
        return False


def decompose_time_series(sales_data, period):
    """
    Decompose the time series data into trend, seasonality, and residuals.
    """
    decomposition = seasonal_decompose(sales_data, model='additive', period=period)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(sales_data, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def main():
    file_path = 'data/IrisInnovations.csv'
    n_days = 365
    title = f'Total Sales for IrisInnovations in the last {n_days} days'

    # Load the data
    last_n_days_data = load_data(file_path, n_days)

    # Plot the time series data
    plot_time_series(last_n_days_data, title)

    # Extract the sales data
    sales_data = last_n_days_data['day_sales_usd'].reset_index(drop=True)
    sales_data.index = range(1, len(sales_data) + 1)
    # save sales data to csv
    sales_data.to_csv('data/sales_data_series.csv', index=False)

    # Check if the data is stationary
    if is_stationary(sales_data):
        # Decompose the time series data
        decompose_time_series(sales_data, period=52)  # Assuming weekly cycles


if __name__ == "__main__":
    main()