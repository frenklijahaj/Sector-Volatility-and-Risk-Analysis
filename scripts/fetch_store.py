import yfinance as yf
import pandas as pd

def fetch_and_store_data(ticker, engine):
    try:
        table_name = ticker.lower()
        
        # Fetch data from Yahoo Finance
        data = yf.download(ticker, start="2010-01-01", end="2025-01-01")
        if data.empty:
            print(f"No data found for {ticker}")
            return

        print(f"Downloaded data for {ticker}")
        
        # Reset index and flatten multiIndex columns
        data.reset_index(inplace=True)
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        # Save data to the database
        data.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"Data for {ticker} saved to the database in table '{table_name}'")
        
    except Exception as e:
        print(f"Error fetching or storing data for {ticker}: {e}")
