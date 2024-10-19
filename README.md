# Rainfall Prediction Project
Created by: Maleke Chaker
Email: melekchaker@gmail.com 

This project involves predicting monthly rainfall using time-series models and a neural network-based approach. We have utilized **Triple Seasonal Exponential Smoothing**, **SARIMA**, and **Artificial Neural Networks (ANN)** to forecast precipitation levels. Below is a step-by-step guide to the project.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Models Used](#models-used)
   - Triple Seasonal Exponential Smoothing
   - SARIMA (Seasonal Auto-Regressive Integrated Moving Average)
   - Artificial Neural Network (ANN)
4. [Evaluation Metrics](#evaluation-metrics)
5. [How to Run](#how-to-run)
6. [Requirements](#requirements)
7. [Results](#results)

## Project Overview

This project aims to predict future rainfall using historical precipitation data. Three main models were implemented:
1. **Triple Seasonal Exponential Smoothing**: A time series smoothing model that accounts for multiple seasonality patterns.
2. **SARIMA Model**: A statistical method that captures autoregressive (AR), differencing (I), and moving average (MA) components along with seasonality.
3. **Artificial Neural Network (ANN)**: A non-linear model built using a feedforward neural network for time series regression.

## Data

### Dataset
The dataset contains monthly measurements of precipitation and weather-related features:
- **Precipitation** (Target variable)
- **Specific Humidity** (Dropped for modeling)
- **Relative Humidity** (Dropped for modeling)
- **Temperature** (Dropped for modeling)

The dataset covers **252 months** of data starting from January 2000 to the end of the period. The primary goal is to predict future precipitation values using previous months' precipitation data.

### Preprocessing
- The irrelevant columns such as **Specific Humidity**, **Relative Humidity**, and **Temperature** were dropped.
- For the **ANN model**, data was normalized using `MinMaxScaler`.

## Models Used

### 1. Triple Seasonal Exponential Smoothing
This model is designed for time series data exhibiting multiple seasonal patterns. We used this model for long-term rainfall forecasting.

#### Metrics for this model:
- **Mean Absolute Error (MAE)**: 110.05
- **Mean Squared Error (MSE)**: 37,350.37
- **Root Mean Squared Error (RMSE)**: 193.26

### 2. SARIMA (Seasonal Auto-Regressive Integrated Moving Average)
SARIMA model captures both autoregressive and moving average behavior in time series data with seasonality. The seasonal component was set to 12-month intervals, and model tuning was performed based on partial autocorrelation plots.

#### Model Configuration:
- **Order**: (1, 0, 0)
- **Seasonal Order**: (2, 0, [1], 12)
- **Fitted with auto_arima** for the best parameter selection.

#### Metrics for SARIMA model:
- **MAE**: 79.57
- **MSE**: 37,989.93
- **RMSE**: 194.91

### 3. Artificial Neural Network (ANN)
A feedforward neural network was used to capture non-linear relationships in the time series data. The architecture includes:
- Input layer: 13 lag features
- One hidden layer: 64 neurons
- Batch normalization and dropout for regularization
- Output layer: 1 neuron (for regression)

Training was performed using **Adam optimizer** and **mean squared error loss**.

#### Model Training:
- **Batch size**: 10
- **Epochs**: Up to 100 (with early stopping)
- **Callbacks**: Early stopping, learning rate reduction, and TensorBoard logging.

#### Metrics for ANN:
- **Final Validation Loss**: ~0.03 (after normalization)

## Evaluation Metrics

To evaluate and compare the models, the following metrics were used:

- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values.
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, which gives a measure in the same unit as the target variable (precipitation).

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rainfall-prediction.git
cd rainfall-prediction
```

### 2. Install Dependencies
Ensure that you have Python installed. You can install the required libraries by running:
```bash
pip install -r requirements.txt
```

### 3. Run the Models
You can run the models individually by executing the corresponding Python files.

- **Triple Seasonal Exponential Smoothing**
  ```bash
  python triple_seasonal_exp_smoothing.py
  ```

- **SARIMA Model**
  ```bash
  python sarima_model.py
  ```

- **ANN Model**
  ```bash
  python ann_model.py
  ```

### 4. Plotting Predictions
Each script contains plotting functions that will generate plots comparing predicted vs actual values for rainfall. The predictions are also printed in the console for each test month.

## Requirements

The main dependencies include:
- **pandas**: For data manipulation.
- **numpy**: For numerical computations.
- **tensorflow**: For building the ANN model.
- **statsmodels**: For the SARIMA model.
- **matplotlib**: For plotting.
- **scikit-learn**: For preprocessing and metrics.

You can install these via the `requirements.txt` file.

## Results

- The **SARIMA model** performed the best overall in terms of MAE and RMSE when compared to other models.
- The **Triple Seasonal Exponential Smoothing model** had higher errors compared to SARIMA.
- The **ANN model** demonstrated promising results with a low validation loss, but further tuning may be required to improve performance in real-world applications.

