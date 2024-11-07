import altair as alt
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
import datetime
import warnings

alt.data_transformers.disable_max_rows()


def calculate_rsi(df, window=10):
    # Calculate daily price changes
    delta = df['Close'].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the average gain and average loss
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_columns(stock_df: pd.DataFrame
                ) -> pd.DataFrame:
    """
    Adds new columns to the stock data DataFrame with various technical indicators.

    Parameters:
    stock_df (DataFrame): DataFrame containing stock data    
    """
    warnings.filterwarnings('ignore')
    print(f'Adding columns...')
    # Create new columns with Returns
    stock_df['1_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(1)) / stock_df['Close'].shift(1) * 100
    stock_df['5_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(5)) / stock_df['Close'].shift(5) * 100
    stock_df['10_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(10)) / stock_df['Close'].shift(10) * 100
    stock_df['20_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(20)) / stock_df['Close'].shift(20) * 100
    stock_df['50_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(50)) / stock_df['Close'].shift(50) * 100
    stock_df['200_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(200)) / stock_df['Close'].shift(200) * 100

    stock_df['Best_Return_Window'] = stock_df[['1_Day_Return', '5_Day_Return',
                                               '10_Day_Return', '20_Day_Return', '50_Day_Return', '200_Day_Return']].idxmax(axis=1)
    stock_df['Best_Return'] = stock_df[['1_Day_Return', '5_Day_Return',
                                        '10_Day_Return', '20_Day_Return', '50_Day_Return', '200_Day_Return']].max(axis=1)
    stock_df['Best_Return_Window'] = stock_df['Best_Return_Window'].replace(
        '_Day_Return', '', regex=True)

    # Create lag columns
    stock_df['close_lag1'] = stock_df['Close'].shift(1)
    stock_df['close_lag2'] = stock_df['Close'].shift(2)
    stock_df['close_lag3'] = stock_df['Close'].shift(3)
    stock_df['close_lag4'] = stock_df['Close'].shift(5)
    stock_df['close_lag5'] = stock_df['Close'].shift(10)

    stock_df['volume_lag1'] = stock_df['Volume'].shift(1)
    stock_df['volume_lag2'] = stock_df['Volume'].shift(2)
    stock_df['volume_lag3'] = stock_df['Volume'].shift(3)
    stock_df['volume_lag4'] = stock_df['Volume'].shift(5)
    stock_df['volume_lag5'] = stock_df['Volume'].shift(10)

    # Create new columns with Moving Averages and Standard Deviations
    stock_df.reset_index(inplace=True)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    # stock_df['Symbol'] = stock_df['St']
    stock_df['MA_10'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=10).mean().reset_index(level=0, drop=True)
    stock_df['MA_20'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=20).mean().reset_index(level=0, drop=True)
    stock_df['MA_50'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=50).mean().reset_index(level=0, drop=True)
    stock_df['MA_200'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=200).mean().reset_index(level=0, drop=True)

    stock_df['std_10'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=10).std().reset_index(level=0, drop=True)
    stock_df['std_20'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=20).std().reset_index(level=0, drop=True)
    stock_df['std_50'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=50).std().reset_index(level=0, drop=True)
    stock_df['std_200'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=200).std().reset_index(level=0, drop=True)

    # Create new columns with Bollinger Bands for each Moving Average
    stock_df['upper_band_10'] = stock_df['MA_10'] + (stock_df['std_10'] * 2)
    stock_df['lower_band_10'] = stock_df['MA_10'] - (stock_df['std_10'] * 2)

    stock_df['upper_band_20'] = stock_df['MA_20'] + (stock_df['std_20'] * 2)
    stock_df['lower_band_20'] = stock_df['MA_20'] - (stock_df['std_20'] * 2)

    stock_df['upper_band_50'] = stock_df['MA_50'] + (stock_df['std_50'] * 2)
    stock_df['lower_band_50'] = stock_df['MA_50'] - (stock_df['std_50'] * 2)

    stock_df['upper_band_200'] = stock_df['MA_200'] + (stock_df['std_200'] * 2)
    stock_df['lower_band_200'] = stock_df['MA_200'] - (stock_df['std_200'] * 2)

    # Create new columns Indicating Golden Cross and Death Cross
    stock_df['Golden_Cross_Short'] = np.where((stock_df['MA_10'] > stock_df['MA_20']) & (
        stock_df['MA_10'].shift(1) <= stock_df['MA_20'].shift(1)), 1, 0)
    stock_df['Golden_Cross_Medium'] = np.where((stock_df['MA_20'] > stock_df['MA_50']) & (
        stock_df['MA_20'].shift(1) <= stock_df['MA_50'].shift(1)), 1, 0)
    stock_df['Golden_Cross_Long'] = np.where((stock_df['MA_50'] > stock_df['MA_200']) & (
        stock_df['MA_50'].shift(1) <= stock_df['MA_200'].shift(1)), 1, 0)

    stock_df['Death_Cross_Short'] = np.where((stock_df['MA_10'] < stock_df['MA_20']) & (
        stock_df['MA_10'].shift(1) >= stock_df['MA_20'].shift(1)), 1, 0)
    stock_df['Death_Cross_Medium'] = np.where((stock_df['MA_20'] < stock_df['MA_50']) & (
        stock_df['MA_20'].shift(1) >= stock_df['MA_50'].shift(1)), 1, 0)
    stock_df['Death_Cross_Long'] = np.where((stock_df['MA_50'] < stock_df['MA_200']) & (
        stock_df['MA_50'].shift(1) >= stock_df['MA_200'].shift(1)), 1, 0)

    # Create new columns for Relative Strength Index (RSI)
    stock_df['RSI_10_Day'] = calculate_rsi(stock_df)
    # Create new columns with Rate of Change and Average Volume

    stock_df['ROC'] = (
        (stock_df['Close'] - stock_df['Close'].shift(1)) / stock_df['Close'].shift(1)) * 100

    stock_df['AVG_Volume_10'] = stock_df.groupby('Symbol')['Volume'].rolling(
        window=10).mean().reset_index(level=0, drop=True)
    stock_df['AVG_Volume_20'] = stock_df.groupby('Symbol')['Volume'].rolling(
        window=20).mean().reset_index(level=0, drop=True)
    stock_df['AVG_Volume_50'] = stock_df.groupby('Symbol')['Volume'].rolling(
        window=50).mean().reset_index(level=0, drop=True)
    stock_df['AVG_Volume_200'] = stock_df.groupby('Symbol')['Volume'].rolling(
        window=200).mean().reset_index(level=0, drop=True)

    print(f'Halfway There...')

    # Doji Candlestick Pattern, identified by a small body and long wicks

    stock_df['EMA_short'] = stock_df['Close'].ewm(span=12, adjust=False).mean()

    # Calculate the long-term EMA
    stock_df['EMA_long'] = stock_df['Close'].ewm(span=26, adjust=False).mean()

    # Calculate the MACD line
    stock_df['MACD'] = stock_df['EMA_short'] - stock_df['EMA_long']

    # Calculate the Signal line
    stock_df['Signal'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate the MACD histogram
    stock_df['MACD_Hist'] = stock_df['MACD'] - stock_df['Signal']

    # Create new columns for Average True Range (ATR) and True Range (TR)

    stock_df['Previous_Close'] = stock_df['Close'].shift(1)

    # True Range, Shows the volatility of the stock
    stock_df['TR'] = stock_df.apply(
        lambda row: max(
            row['High'] - row['Low'],  # High - Low
            # |High - Previous Close|
            abs(row['High'] - row['Previous_Close']),
            abs(row['Low'] - row['Previous_Close'])  # |Low - Previous Close|
        ), axis=1
    )

    # Average True Range, Shows the average volatility of the stock
    stock_df['ATR'] = stock_df['TR'].rolling(window=10).mean()

    # Create new columns for Relative Strength Index (RSI) and Rate of Change (ROC)
    stock_df['10_Day_ROC'] = (
        (stock_df['Close'] - stock_df['Close'].shift(10)) / stock_df['Close'].shift(10)) * 100
    stock_df['20_Day_ROC'] = (
        (stock_df['Close'] - stock_df['Close'].shift(20)) / stock_df['Close'].shift(20)) * 100
    stock_df['50_Day_ROC'] = (
        (stock_df['Close'] - stock_df['Close'].shift(50)) / stock_df['Close'].shift(50)) * 100

    # Create new columns for 10,20,50 day resistance and support levels
    stock_df['Resistance_10_Day'] = stock_df['Close'].rolling(window=10).max()
    stock_df['Support_10_Day'] = stock_df['Close'].rolling(window=10).min()
    stock_df['Resistance_20_Day'] = stock_df['Close'].rolling(window=20).max()
    stock_df['Support_20_Day'] = stock_df['Close'].rolling(window=20).min()
    stock_df['Resistance_50_Day'] = stock_df['Close'].rolling(window=50).max()
    stock_df['Support_50_Day'] = stock_df['Close'].rolling(window=50).min()

    # Create new columns for 10,20,50 day Volume Indicators
    stock_df['Volume_MA_10'] = stock_df['Volume'].rolling(window=10).mean()
    stock_df['Volume_MA_20'] = stock_df['Volume'].rolling(window=20).mean()
    stock_df['Volume_MA_50'] = stock_df['Volume'].rolling(window=50).mean()
    # Use a smoothed version of 'Close' to detect peaks and troughs
    stock_df['Smoothed_Close'] = stock_df['Close'].rolling(window=20).mean()

    # Find local minima (buy points) and local maxima (sell points)
    # Local minima (buy points)
    stock_df['Buy_Signal'] = (stock_df['Smoothed_Close'].shift(1) > stock_df['Smoothed_Close']) & (
        stock_df['Smoothed_Close'].shift(-1) > stock_df['Smoothed_Close'])

    # Local maxima (sell points)
    stock_df['Sell_Signal'] = (stock_df['Smoothed_Close'].shift(1) < stock_df['Smoothed_Close']) & (
        stock_df['Smoothed_Close'].shift(-1) < stock_df['Smoothed_Close'])

    # Initialize 'Optimal_Action' column with 'Hold'
    stock_df['Optimal_Action'] = 'Hold'

    # Assign 'Buy' where Buy_Signal is True
    stock_df.loc[stock_df['Buy_Signal'], 'Optimal_Action'] = 'Buy'

    # Assign 'Sell' where Sell_Signal is True
    stock_df.loc[stock_df['Sell_Signal'], 'Optimal_Action'] = 'Sell'

    # Clean up: drop the temporary signals if needed
    stock_df.drop(['Buy_Signal', 'Sell_Signal',
                  'Smoothed_Close'], axis=1, inplace=True)

    stock_df['Z-score'] = (stock_df['Close'] -
                           stock_df['Close'].mean()) / stock_df['Close'].std()

    stock_df.fillna(0, inplace=True)

    stock_df['OBV'] = 0
    for i in range(1, len(stock_df)):
        if stock_df['Close'].iloc[i] > stock_df['Close'].iloc[i - 1]:
            stock_df.loc[stock_df.index[i],
                         'OBV'] = stock_df['OBV'].iloc[i - 1] + stock_df['Volume'].iloc[i]
        elif stock_df['Close'].iloc[i] < stock_df['Close'].iloc[i - 1]:
            stock_df.loc[stock_df.index[i],
                         'OBV'] = stock_df['OBV'].iloc[i - 1] - stock_df['Volume'].iloc[i]
        else:
            stock_df.loc[stock_df.index[i],
                         'OBV'] = stock_df['OBV'].iloc[i - 1]

    return stock_df


def plot_closing_price(df: pd.DataFrame,
                       opacity: int = 1,
                       window: list = ['2024-01-01', '2024-10-01']
                       ) -> alt.Chart:
    """
    Plots the closing price of a stock over time.

    Parameters:
    df (DataFrame): DataFrame containing stock data
    company (str): Company stock symbol
    opacity (float): Opacity of the line plot

    Returns:
    chart (altair.Chart): Altair line chart of the closing
    """
    company = df['Symbol'].iloc[0]    
    df['Date'] = pd.to_datetime(df['Date'])
    start = pd.to_datetime(window[0])
    stop = pd.to_datetime(window[1])
    df = df[df['Date'] >= start]
    df = df[df['Date'] <= stop]

    chart = alt.Chart(df).mark_line(color='black', opacity=opacity).encode(
        alt.X('Date:T', title='Date'),
        alt.Y('Close:Q', title='Closing Price'),
        alt.Tooltip(['Date:T', 'Close:Q', 'Volume:Q']),
    ).properties(
        title=f'{company} Closing Price',
        width=800,
        height=400
    ).interactive(
        bind_y=False
    )
    return chart


def plot_candlestick(df: pd.DataFrame,
                     window: list = ['2023-01-01', '2024-10-01'],
                     SR_window=None) -> go.Figure:
    """
    Plots a Candlestick Chart for a given stock symbol and date range.

    Parameters:
    df (DataFrame): DataFrame containing stock data
    company (str): Company stock symbol
    window (list): Date range to filter the data, 
                     default is from '2023-01-01' to '2024-10-01'
    SR_window (int): Support and Resistance window

    Returns:
    fig (plotly.graph_objects.Figure): Candlestick chart

    """
    # Fileter data to specified stock and date range
    company = df['Symbol'].iloc[0]
    df['Date'] = pd.to_datetime(df['Date'])
    start = pd.to_datetime(window[0])
    stop = pd.to_datetime(window[1])
    df = df[df['Date'] >= start]
    df = df[df['Date'] <= stop]

    # Create the Candlesick Chart
    candlestick = go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    )

    # Create the Volume Bar graph
    volume = go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        marker_color='blue',
        opacity=0.5,
        yaxis='y2'
    )
    if SR_window:
        # Create the Support and Resistance Lines
        support = go.Scatter(
            x=df['Date'],
            y=df[f'Support_{SR_window}_Day'],
            mode='lines',
            name=f'Support {SR_window} Day',
            line=dict(color='green', width=1)
        )

        resistance = go.Scatter(
            x=df['Date'],
            y=df[f'Resistance_{SR_window}_Day'],
            mode='lines',
            name=f'Resistance {SR_window} Day',
            line=dict(color='red', width=1)
        )

    # Create the Layout for the Chart (Title, Axis Labels, etc.)
    layout = go.Layout(
        title=f'{company} Candlestick Chart From {window[0]} to {window[1]}',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price', showgrid=True),
        yaxis2=dict(title='Volume', overlaying='y',
                    side='right', showgrid=False),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',  # Compare data points on hover
        plot_bgcolor='white'
    )

    # Layer all the charts together into one figure
    fig = go.Figure(
        data=[candlestick, volume, support, resistance],
        layout=layout
    )
    return fig


def plot_moving_average(df: pd.DataFrame,
                        MA: int,
                        color: str = 'blue',
                        term: str = None,
                        window: list = ['2024-01-01', '2024-10-01']
                        ) -> alt.Chart:
    """
    Plots the moving average of a stock over time.

    Parameters:
    df (DataFrame): DataFrame containing stock data
    company (str): Company stock symbol
    color (str): Color of the line plot
    MA (str): Moving Average window
    term (str): Term of the moving average (Long Term, Short Term)

    Returns:
    chart (altair.Chart): Altair line chart of the moving average
    """
    df['Date'] = pd.to_datetime(df['Date'])
    start = pd.to_datetime(window[0])
    stop = pd.to_datetime(window[1])
    df = df[df['Date'] >= start]
    df = df[df['Date'] <= stop]

    company = df['Symbol'].iloc[0]
    chart = alt.Chart(df).mark_line(color=color).encode(
        alt.X('Date:T', title='Date'),
        alt.Y(f'MA_{MA}:Q', title='{MA} Day Moving Average'),
        alt.Tooltip([f'MA_{MA}:Q', 'Date', 'Close:Q', 'Volume:Q', 'Death_Cross_Short:O',
                    'Golden_Cross_Short:O', 'Death_Cross_Long:O', 'Golden_Cross_Long:O'])
    ).properties(
        title=f'{company} Moving Average {term}',
        width=800,
        height=400
    ).interactive(
        bind_y=False
    )

    return chart


def plot_bollinger_bands(df: pd.DataFrame,
                         band: str,
                         window: list = ['2024-01-01', '2024-10-01']
                         ) -> go.Figure:
    """
    Plots the bollinger bands of a stock over time.

    Parameters:
    df (DataFrame): DataFrame containing stock data
    band (str): Band to plot (10, 20, 50, 200)
    window (list): Date range to filter the data, 
                     default is from '2023-01-01' to '2024-10-01'

    Returns:
    fig (plotly.graph_objects.Figure): Bollinger Bands chart
    """
    company = df['Symbol'].iloc[0]

    # Filter data to specified stock and date range
    df['Date'] = pd.to_datetime(df['Date'])
    start = pd.to_datetime(window[0])
    stop = pd.to_datetime(window[1])
    df = df[df['Date'] >= start]
    df = df[df['Date'] <= stop]
    fig = go.Figure()

    # Create the Close Price Line Chart
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Close Price'
        )
    )

    # Create the upper band
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df[f'upper_band_{band}'],
            mode='lines',
            name='Upper Band',
            line=dict(color='red', width=1)
        )
    )

    # Create the lower band
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df[f'lower_band_{band}'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.2)',
            name='Lower Band',
            line=dict(color='red', width=1)
        )
    )

    # Update layout
    fig.update_layout(
        title=f'{company} Bollinger Bands for {band} Day Moving Average',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        width=1000,
        height=600
    )

    fig.show()


def plot_RSI(df: pd.DataFrame,
             start: str = '2024-01-01'
             ) -> plt.subplot:
    """
    Plots the Relative Strength Index (RSI) of a stock over time.

    Parameters:
    df (DataFrame): DataFrame containing stock data
    company (str): Company stock symbol
    start (str): Start date to filter the data

    Returns:
    plt (plt.subplot): RSI chart
    """
    start = pd.to_datetime(start)
    df = df[df['Date'] >= start]
    company = df['Symbol'].iloc[0]
    fig, axs = plt.subplots(
        2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))

    axs[0].set_title(f'{company} Closing Price')
    axs[0].plot(df['Date'], df['Close'], color='black')
    axs[1].axhline(y=70, color='red', linestyle='--')
    axs[1].axhline(y=30, color='green', linestyle='--')
    axs[1].plot(df['Date'], df['RSI_10_Day'], color='orange')
    axs[1].set_title('RSI')
    plt.show()

    return plt


def plot_MACD(df: pd.DataFrame,
              window: list = ['2024-01-01', '2024-10-01']
              ) -> go.Figure:
    """
    Plots the Moving Average Convergence Divergence

    Parameters:
    df (DataFrame): DataFrame containing stock data
    company (str): Company stock symbol
    window (list): Date range to filter the data, 
                        default is from '2023-01-01' to '2024-10-01'

    Returns:
    fig (plotly.graph_objects.Figure): Bollinger Bands chart

    """
    company = df['Symbol'].iloc[0]
    df = df[df['Symbol'] == company]
    df['Date'] = pd.to_datetime(df['Date'])
    start = pd.to_datetime(window[0])
    stop = pd.to_datetime(window[1])
    df = df[df['Date'] >= start]
    df = df[df['Date'] <= stop]

    # Create a Plotly figure with two subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Close Price', 'MACD')
    )

    # Add Close Price graph
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines',
        name='Close Price'
    ), row=1, col=1)

    # Add MACD line graph
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='red')
    ), row=2, col=1)

    # Add Signal line graph
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Signal'],
        mode='lines',
        name='Signal',
        line=dict(color='blue')
    ), row=2, col=1)

    # Add MACD histogram
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['MACD_Hist'],
        name='MACD Histogram',
        marker_color=['green' if x > 0 else 'red' for x in df['MACD_Hist']]
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        title=f'MACD for {company}',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        width=1000,
        height=600
    )

    fig.show()


def get_stock_data(symbol: str
                   ) -> pd.DataFrame:
    """
    Gets the stock data for a given symbol and adds new columns to the DataFrame.

    Parameters:
    symbol (str): Stock symbol to get data for
    """
    start_date = '2010-01-01'
    stock_df = yf.download(symbol, start=start_date)
    stock_df['Symbol'] = symbol
    stock_df.reset_index(inplace=True)
    if (stock_df['Date'].tail(1).values != datetime.datetime.now().strftime('%Y-%m-%d')):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period='1d', interval='1d')
        except:
            stock = yf.Ticker(symbol)

            data = stock.history(period='1d', interval='1d')

        if not data.empty:
            latest_data = data.iloc[-1]
            time = latest_data.name
            open_price = latest_data['Open']
            high = latest_data['High']
            low = latest_data['Low']
            close = latest_data['Close']
            volume = latest_data['Volume']
            new_row = pd.DataFrame({
                'Symbol': [symbol],
                'Date': [datetime.datetime.strftime(time, '%Y-%m-%d')],
                'Open': [open_price],
                'High': [high],
                'Low': [low],
                'Close': [close],
                'Volume': [volume]
            })

            new_row = new_row.reset_index(drop=True)

            stock_df = pd.concat([stock_df, new_row],
                                 ignore_index=True).fillna(0)
    return stock_df
