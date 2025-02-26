import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scripts.analyze import process_data

def process_data_for_ml(ticker, engine):
    try:
        # Load processed data and the name of the volatility column
        data, volatility_col = process_data(ticker, engine)
        if data is None or data.empty:
            return None, None

        # Convert 'Date' to datetime, sort and reset index
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data.sort_values('Date', inplace=True)
            data.reset_index(drop=True, inplace=True)

        # Create a trend feature as a simple time index
        data['Trend'] = np.arange(len(data))

        # Create seasonal features based on the month
        data['Month'] = data['Date'].dt.month
        data['sin_month'] = np.sin(2 * np.pi * data['Month'] / 12)
        data['cos_month'] = np.cos(2 * np.pi * data['Month'] / 12)

        # Create lag features for volatility
        data['Lag_1'] = data[volatility_col].shift(1)
        data['Lag_2'] = data[volatility_col].shift(2)

        # Define features and target (predicting next-day volatility)
        features = ['Lag_1', 'Lag_2', 'Trend', 'sin_month', 'cos_month']
        X = data[features]
        y = data[volatility_col].shift(-1)

        # Drop rows with missing values
        valid_indices = y.dropna().index
        return X.loc[valid_indices], y.loc[valid_indices]

    except Exception as e:
        print(f"Error processing data for ML for {ticker}: {e}")
        return None, None


def train_regression_model(ticker, engine):
    try:
        X, y = process_data_for_ml(ticker, engine)
        if X is None or y is None:
            print(f"No data available for regression training on {ticker}")
            return None

        # Split data into training and testing sets (without shuffling to preserve time order)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Best parameters per ticker (Customize as needed - Gotten from hyperparameter tuning)
        best_params = {
            'XLK': {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.9},
            'XLF': {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.6, 'colsample_bytree': 0.6},
            'XLE': {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.9},
            'XLU': {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.9},
            'XLRE': {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.9},
        }
        params = best_params.get(ticker, {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 1.0})
        
        # Initialize and train the XGBoost Regressor
        model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            random_state=42,
            objective='reg:squarederror'
        )
        model.fit(X_train, y_train)

        # Evaluate the model on the test set
        y_pred = model.predict(X_test)
        print(f"XGBoost Regression Evaluation for {ticker}:")
        print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
        print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))

        return model
    except Exception as e:
        print(f"Error training regression model for {ticker}: {e}")
        return None


def forecast_volatility(historical_data, model, start_date, end_date):
    try:
        # Generate future business dates
        future_dates = pd.date_range(start=start_date, end=end_date, freq='B')

        # Use the last two observed volatility values from historical data
        if 'Volatility' not in historical_data.columns:
            print("Historical data must include a 'Volatility' column.")
            return None
        last_volatility = historical_data['Volatility'].iloc[-1]
        second_last_volatility = historical_data['Volatility'].iloc[-2] if len(historical_data) > 1 else last_volatility

        # Get the last trend value (if already computed) or use the index length
        last_trend = historical_data['Trend'].iloc[-1] if 'Trend' in historical_data.columns else len(historical_data) - 1

        forecast_results = []
        # Initialize current lag values
        current_lag1 = last_volatility
        current_lag2 = second_last_volatility

        # Recursive forecasting using the ML model predictions
        for i, date in enumerate(future_dates, start=1):
            forecast_trend = last_trend + i
            # Compute seasonal features for the forecast date
            month = date.month
            sin_month = np.sin(2 * np.pi * month / 12)
            cos_month = np.cos(2 * np.pi * month / 12)

            # Build the feature vector
            input_features = pd.DataFrame([{
                'Lag_1': current_lag1,
                'Lag_2': current_lag2,
                'Trend': forecast_trend,
                'sin_month': sin_month,
                'cos_month': cos_month
            }])

            # Predict next-day volatility using the trained model
            predicted_volatility = model.predict(input_features)[0]
            predicted_volatility = max(0, predicted_volatility)  

            # Store the forecast
            forecast_results.append({
                'Date': date,
                'Volatility': predicted_volatility
            })

            # Update lag features for the next iteration
            current_lag2 = current_lag1
            current_lag1 = predicted_volatility

        forecast_df = pd.DataFrame(forecast_results)
        return forecast_df

    except Exception as e:
        print(f"Error forecasting volatility for future dates: {e}")
        return None


def save_model(model, model_path):
    try:
        joblib.dump(model, model_path)
        print(f"Model saved at {model_path}")
    except Exception as e:
        print(f"Failed to save model: {e}")


def load_model(model_path):
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def predict_future_volatility(ticker, engine, model_path, start_date, end_date):
    try:
        # Load historical data 
        historical_data, volatility_col = process_data(ticker, engine)
        if historical_data is None or historical_data.empty:
            print(f"No historical data available for {ticker}. Skipping future volatility prediction.")
            return None

        if 'Volatility' not in historical_data.columns:
            print(f"Historical data for {ticker} must contain a 'Volatility' column.")
            return None

        # Load the trained model
        model = load_model(model_path)
        if not model:
            print(f"No trained model found for {ticker}. Skipping future volatility prediction.")
            return None

        # Forecast future volatility using the model
        forecast_df = forecast_volatility(historical_data, model, start_date, end_date)
        if forecast_df is None or forecast_df.empty:
            print(f"Failed to forecast volatility for {ticker}.")
            return None

        print(f"Future Volatility Predictions for {ticker}:\n", forecast_df.head())

        return forecast_df

    except Exception as e:
        print(f"Error predicting future volatility for {ticker}: {e}")
        return None