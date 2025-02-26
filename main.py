from scripts.fetch_store import fetch_and_store_data
from scripts.graphs import graph_data, compare_data, data_with_predictions
from scripts.ml_model import train_regression_model, save_model, predict_future_volatility
from scripts.analyze import process_data
from sqlalchemy import create_engine
import os

def main():
    # Defining tickers and database name
    tickers = ["XLK", "XLF", "XLE", "XLU", "XLRE"]  # Technology, Financials, Energy, Utilities, Real Estate
    database_name = "sector_data"

    # Connection to database
    password = os.getenv("DB_PASSWORD")
    if not password:
        raise ValueError("Database password not set in environment variables.")
    
    engine = create_engine(f"postgresql+psycopg2://postgres:{password}@localhost/{database_name}")

    try:
        # Fetch and store data
        for ticker in tickers:
            fetch_and_store_data(ticker, engine)

        # Analyze and graph data
        for ticker in tickers:
            processed_data = process_data(ticker, engine)
            if processed_data is not None:
                graph_data(ticker, engine)
        compare_data(tickers, engine)
        
        # Train and save models
        for ticker in tickers:
            model = train_regression_model(ticker, engine)
            if model:
                model_path = f"models/{ticker}_regression_model.pkl"
                save_model(model, model_path)
                
        # Predict future volatility using models
        for ticker in tickers:
            model_path = f"models/{ticker}_regression_model.pkl"
            forecast_data = predict_future_volatility(
                ticker, engine, model_path, start_date="2025-01-01", end_date="2026-12-31"
            )
            if forecast_data is not None:
                # Visualize GARCH predictions alongside raw historical data
                data_with_predictions(ticker, engine, synthetic_data=forecast_data)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
