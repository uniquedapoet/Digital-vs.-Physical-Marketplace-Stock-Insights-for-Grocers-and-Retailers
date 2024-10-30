import altair as alt
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import yfinance as yf
import datetime


alt.data_transformers.disable_max_rows()


def plot_closing_price(df, company, opacity=1):
    filtered_df = df[df['Symbol'] == company]
    chart = alt.Chart(filtered_df).mark_line(color='black', opacity=opacity).encode(
        alt.X('Date:T', title='Date'),
        alt.Y('Close:Q', title='Closing Price'),
        alt.Tooltip(['Date:T', 'Close:Q', 'Volume:Q', 'RSI_10_Day']),
    ).properties(
        title=f'{company} Closing Price',
        width=800,
        height=400
    ).interactive(
        bind_y=False
    )
    return chart


def plot_candlestick(df, company,window:list = None,SR_window=10):
    # Fileter data to specified stock and date range
    filtered_df = df[df['Symbol'] == company]
    filtered_df = filtered_df[filtered_df['Date'] >= window[0]]
    filtered_df = filtered_df[filtered_df['Date'] <= window[1]]
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

    df = filtered_df

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
        title= f'{company} Candlestick Chart From {window[0]} to {window[1]}',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price', showgrid=True),
        yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
        xaxis_rangeslider_visible=False,
        hovermode='x unified', # Compare data points on hover
        plot_bgcolor='white'
    )

    # Layer all the charts together into one figure
    fig = go.Figure(
        data=[candlestick, volume, support, resistance],
        layout=layout
        )
    return fig


def plot_moving_average(df, company, color, MA,term):
    filtered_df = df[df['Symbol'] == company]
    chart = alt.Chart(filtered_df).mark_line(color=color).encode(
        alt.X('Date:T', title='Date'),
        alt.Y(f'{MA}:Q', title='{MA}'),
        alt.Tooltip([MA,'Date', 'Close:Q', 'Volume:Q', 'Death_Cross_Short:O', 'Golden_Cross_Short:O', 'Death_Cross_Long:O', 'Golden_Cross_Long:O'])
    ).properties(
        title=f'{company} Moving Average {term}',
        width=800,
        height=400
    ).interactive(
        bind_y=False
    )

    
    return chart


def plot_bollinger_bands(df, company, band):
    # Filter data to specified stock and date range
    df = df[df['Symbol'] == company]
    window=['2023-01-01', '2024-10-01']
    df = df[df['Date'] >= window[0]]
    df = df[df['Date'] <= window[1]]
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
        title=f'Bollinger Bands for {band} Day Moving Average',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        width=1000,
        height=600
    )

    fig.show()


def plot_RSI(df, company,start):
    filtered_df = df[df['Symbol'] == company]
    filtered_df = filtered_df[filtered_df['Date'] >= start]
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))
    
    axs[0].set_title(f'{company} Closing Price')
    axs[0].plot(filtered_df['Date'],filtered_df['Close'], color='black')
    axs[1].axhline(y=70, color='red', linestyle='--')
    axs[1].axhline(y=30, color='green', linestyle='--')
    axs[1].plot(filtered_df['Date'],filtered_df['RSI_10_Day'], color='orange')
    axs[1].set_title('RSI')
    plt.show()


def plot_MACD(df, company, start, stop):
    
    # Filter the DataFrame for the specified company and date range
    df = df[(df['Symbol'] == company)] 
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