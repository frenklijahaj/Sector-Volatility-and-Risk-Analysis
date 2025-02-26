import pandas as pd
import matplotlib.pyplot as plt

def graph_data(ticker, engine):
    try:
        # Get processed data from the database
        table_name = ticker.lower()
        query = f"SELECT \"Date\", \"Close\", \"Returns\", \"Volatility\", \"High_Volatility\" FROM {table_name}"
        processed_data = pd.read_sql(query, engine)

        if processed_data.empty:
            print(f"No data available for analysis for {ticker}.")
            return

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(processed_data['Date'], processed_data['Volatility'], label="30-Day Rolling Std (Volatility)", alpha=0.7)
        plt.fill_between(
            processed_data['Date'],
            processed_data['Volatility'],
            where=processed_data['High_Volatility'],
            color='red',
            alpha=0.3,
            label="High Volatility"
        )
        plt.title(f"{ticker} Rolling Volatility")
        plt.xlabel("Date")
        plt.ylabel("Volatility (%)")
        plt.legend()
        plt.grid()
        plt.show()

        print(f"Graphing complete for {ticker}.")
    except Exception as e:
        print(f"Error analyzing data for {ticker}: {e}")

def compare_data(tickers, engine):
    try:
        plt.figure(figsize=(12, 8))

        for ticker in tickers:
            # Fetch processed data from the database
            table_name = ticker.lower()
            query = f"SELECT \"Date\", \"Volatility\" FROM {table_name}"
            processed_data = pd.read_sql(query, engine)

            if processed_data.empty:
                print(f"No data available for {ticker}. Skipping.")
                continue

            # Plot volatility for the ticker
            plt.plot(processed_data['Date'], processed_data['Volatility'], label=f"{ticker} Volatility", alpha=0.7)

        plt.title("Rolling Volatility Comparison Across Sectors")
        plt.xlabel("Date")
        plt.ylabel("Volatility (%)")
        plt.legend()
        plt.grid()
        plt.show()

        print(f"Comparison analysis complete for tickers: {', '.join(tickers)}")
    except Exception as e:
        print(f"Error comparing data: {e}")

def data_with_predictions(ticker, engine, synthetic_data=None):
    try:
        # Fetch processed data from the database
        table_name = ticker.lower()
        query = f"SELECT \"Date\", \"Volatility\" FROM {table_name}"
        processed_data = pd.read_sql(query, engine)

        if processed_data.empty:
            print(f"No data available for {ticker}.")
            return

        # Ensure 'Date' is in datetime format
        processed_data['Date'] = pd.to_datetime(processed_data['Date'])
        processed_data.sort_values(by='Date', inplace=True)

        # Plot historical volatility
        plt.figure(figsize=(10, 6))
        plt.plot(
            processed_data['Date'],
            processed_data['Volatility'],
            label="30-Day Rolling Std (Volatility - Historical)",
            color='blue',
            alpha=0.7
        )

        # Add synthetic data if available
        if synthetic_data is not None:
            # Ensure 'Date' is in datetime format
            synthetic_data['Date'] = pd.to_datetime(synthetic_data['Date'])
            synthetic_data.sort_values(by='Date', inplace=True)

            # Plot synthetic volatility
            plt.plot(
                synthetic_data['Date'],
                synthetic_data['Volatility'],
                label="30-Day Rolling Std (Volatility - Future)",
                color='orange',
                linestyle='--',
                alpha=0.7
            )

        # Add titles, labels, and legend
        plt.title(f"{ticker} Rolling Volatility with Predictions")
        plt.xlabel("Date")
        plt.ylabel("Volatility (%)")
        plt.legend()
        plt.grid()
        plt.show()

        print(f"Prediction graph complete for {ticker}.")

    except Exception as e:
        print(f"Error generating prediction graph for {ticker}: {e}")

