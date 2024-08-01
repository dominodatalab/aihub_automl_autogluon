# Sales forecasting

## About this project
This project shows how to use [Autogluon](https://auto.gluon.ai/stable/index.html) to build regression models to make sales forecasts. The dataset that was used contains data for 5 different SKUs and a model was built for each of the SKU's. Typically for such use cases time series models are applied but based on prior experiments on this dataset, converting the problem to a regression problem and applying regression models gave better results.

The assets available in this project are:
*/AutogluonModels/regression/* - the folders here have the models that were built using Autogluon with additional features from the feature engineering phase. Given the size, the `Atlantis_IrisInnovations` model is not in the repo.

*data/feature_engineering/* - the files in this folder are the final files that contain all the original and derived features for the different `series_id`; these files can be used to build the models

*data/train_test/* - the files in this folder are the train-evaluation sets that were used to train the models for the different `series_id` using Autogluon

*data/obfuscated_6_series_2021-2024_billing_and_holidays.csv* - the data that was originally supplied to build the models

*data/Atlantis_IrisInnovations.csv* - data subset from `data/obfuscated_6_series_2021-2024_billing_and_holidays.csv` that only has data for `Atlantis_IrisInnovations`

*data/Atlantis_OptiGlimpse.csv* - data subset from `data/obfuscated_6_series_2021-2024_billing_and_holidays.csv` that only has data for `Atlantis_OptiGlimpse`

*data/Narnia_LensLogic.csv* - data subset from `data/obfuscated_6_series_2021-2024_billing_and_holidays.csv` that only has data for `Narnia_LensLogic`

*data/Narnia_OptiGlimpse.csv* - data subset from `data/obfuscated_6_series_2021-2024_billing_and_holidays.csv` that only has data for `Narnia_OptiGlimpse`

*data/Rivendell_SightSphere.csv* - data subset from `data/obfuscated_6_series_2021-2024_billing_and_holidays.csv` that only has data for `Rivendell_SightSphere`

*data/prediction_samples* - this folder has examples of data that be used for the job to get predictions

*plots/autogluon_no_feature_engineering* - this folder has some sample forecasts from the `Autogluon` models that were build using time series models and without any derived features. 

*reports/* - this folder has the training results of the models that were built using AutoGluon regression models using the original and the derived features. These models are the ones that are deployed and that performed better than DataRobot's models

*autogluon_forecast_no_feature_eng.py* - This file has code to build Autogluon time series models only using the data that was supplied, it does not use and derived features and hence the performance is not as good as the the Autogluon regression models

*autogluon_regression.py* - This file has code to build Autogluon regression models that use the data that was supplied as well as derived features. Note that all the derived features were not used to build the models and some were dropped before the `fit` method is called

*create_series_subset.py* - This is a helper file that creates csv's for a given `series_id`

*eda.py* - This file has code to do some exploratory time series analysis namely plotting the time series, checking for stationarity, decomposing and visualizing the trend, seasonality, volatility.

*eval_model_metrics.py* - This file contains code to load Autogluon models, datasets to produce model performance reports and to get different performance metrics (RMSE, MAPE, MAE, SMAPE) for the Autogluon models. 

*feature_engineering.py* - This file contains code to derive features from the subsets of the data that correspond to the `series_id`

*invoke_model.py* - This file contains code to call the model API to get predictions

*model.py* - This file contains code to deploy the built models as a model API. It can also be used to call a job. The `predict_with_models` functions it the function that gets called from the model API and the arguments specified in `parser` in the `if __name__ == "__main__":` predicate are used to run the job. Note, in order to run this as a job the `--is_job` must be specified as a CLI argument 


## Set up instructions

This project requires the following [compute environments](https://docs.dominodatalab.com/en/latest/user_guide/f51038/environments/) to be present. Please ensure the "Automatically make compatible with Domino" checkbox is selected while creating the environment.

### Environment Requirements

**Environment Base**
`5.9 Domino Standard Environment Py3.9 R4.3.1 - Revision #1`

***Dockerfile instructions***
```
RUN pip install FLAML==2.1.2 \
	autogluon==1.1.1 \
	lightgbm==4.3.0 \
	lightning==2.2.5 \
	lightning-utilities==0.8.0 \
	matplotlib \
	numba \
	numpy==1.26.4 \
	pandas \
	python-dotenv==1.0.1 \
	pytorch-forecasting \
	pytorch-lightning \
	pytorch_optimizer==2.12.0 \
	pytz==2023.3 \
	scikit-learn \
	scipy \
	seaborn \
	statsmodels \
	torch \
	tqdm==4.66.4
```

### Hardware Requirements
The model training, job and model API used the `large (6 core, 27GB RAM)` hardware tier. The `Workspace and Jobs Volume Size` setting of the workspace was set to 30GB. The project can be run with lower resources but we overprovisioned just to make sure we don't run into any resource constraints when dealing with large models and datasets.

