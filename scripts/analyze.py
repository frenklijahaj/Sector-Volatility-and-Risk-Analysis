import pandas as pd

def process_data(ticker, engine):
    try:
        table_name = ticker.lower()
        query = f"SELECT \"Date\", \"Close\" FROM {table_name}" 
        data = pd.read_sql(query, engine)

        if data.empty:
            print(f"No data found for {ticker}")
            return None, None
        
        # Ensure Date is present and sort by it
        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values(by='Date', inplace=True)

        # Calculate daily returns
        data['Returns'] = data['Close'].pct_change()

        # Ensure sufficient data before rolling calculation
        if len(data) < 30:
            print(f"Not enough data for rolling calculations. Rows available: {len(data)}.")
            return None, None

        # Calculate Volatility
        volatility_col = "Volatility"
        data[volatility_col] = data['Returns'].rolling(window=30).std() * 100
        
        # Add High Volatility column for ML tasks
        data['High_Volatility'] = data[volatility_col] > 2 
        
        # Append the processed data with new columns back to SQL 
        data.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"Processed data for {ticker} and updated SQL table.")

        return data, volatility_col

    except Exception as e:
        print(f"Error processing data for {ticker}: {e}")
        return None, None
    