# 1. Import Libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# 2. Load Your Data
df = pd.read_csv(r"C:\Users\chauh\Downloads\cleaned_sensex.csv", encoding="latin1")

# Data processing
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

# Calculate Daily Return and Crash Detection
df['Daily_Return'] = df['Close'].pct_change() * 100
crash_threshold_daily = -5
df['Crash_Daily'] = df['Daily_Return'] <= crash_threshold_daily

# Create Plot for Sensex Closing Price and Daily Crashes
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))

crash_days = df.index[df['Crash_Daily']]
crash_closes = df['Close'][df['Crash_Daily']]
fig1.add_trace(go.Scatter(x=crash_days, y=crash_closes, mode='markers',
                          marker=dict(color='red'), name=f'Daily Crash (<= {crash_threshold_daily}%)'))

fig1.update_layout(
    title='Sensex Closing Price with Daily Crashes Highlighted',
    xaxis_title='Date',
    yaxis_title='Sensex Close',
    xaxis_rangeslider_visible=True,
    template="plotly_white"
)

# Calculate Market Drawdown
df['Cumulative_Max'] = df['Close'].cummax()
df['Drawdown'] = (df['Close'] - df['Cumulative_Max']) / df['Cumulative_Max'] * 100
drawdown_threshold = -20

# Create Drawdown Plot
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index, y=df['Drawdown'], mode='lines', name='Drawdown (%)'))
fig2.add_hline(y=drawdown_threshold, line_dash='dash', line_color='red',
               annotation_text=f'Drawdown Threshold ({drawdown_threshold}%)', annotation_position='bottom right')
fig2.update_layout(
    title='Market Drawdown Over Time',
    xaxis_title='Date',
    yaxis_title='Drawdown (%)',
    xaxis_rangeslider_visible=True,
    template="plotly_white"
)

# Early Warning System for 2025
np.random.seed(42)
dates_2025 = pd.bdate_range(start="2025-01-01", periods=250)
daily_returns = np.zeros(250)
daily_returns[:150] = np.random.normal(loc=0.0005, scale=0.01, size=150)
daily_returns[150:200] = np.random.normal(loc=-0.008, scale=0.025, size=50)
daily_returns[200:] = np.random.normal(loc=0.0005, scale=0.01, size=50)

prices = [30000]
for ret in daily_returns:
    prices.append(prices[-1]*(1+ret))
prices = prices[1:]

df_2025 = pd.DataFrame({
    'Date': dates_2025,
    'Close': prices,
    'Daily_Return': daily_returns * 100 
})
df_2025.set_index('Date', inplace=True)
df_2025['Rolling_Mean_Return'] = df_2025['Daily_Return'].rolling(window=10).mean()
df_2025['Rolling_Volatility'] = df_2025['Daily_Return'].rolling(window=10).std()

warning_condition = (df_2025['Rolling_Mean_Return'] < -0.5) & (df_2025['Rolling_Volatility'] > 2)
df_2025['Warning'] = warning_condition

# Create Early Warning Signal Plot
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df_2025.index, y=df_2025['Close'], mode='lines', name='Closing Price'))
warning_dates = df_2025.index[df_2025['Warning']]
warning_closes = df_2025['Close'][df_2025['Warning']]
fig3.add_trace(go.Scatter(x=warning_dates, y=warning_closes, mode='markers',
                          marker=dict(color='red'), name='Early Warning Signal'))
fig3.update_layout(
    title='Synthetic 2025 Sensex Closing Price with Early Warning Signals',
    xaxis_title='Date',
    yaxis_title='Sensex Close',
    xaxis_rangeslider_visible=True, 
    template="plotly_white"
)

# 3. Predictive Model - Random Forest Regressor
df['DayOfYear'] = df.index.dayofyear
X = df[['DayOfYear']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 4. Create Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Stock Market Dashboard", style={'textAlign': 'center', 'fontFamily': 'Arial'}),

    # Row for Graphs
    html.Div([
        html.Div([
            html.H2("Sensex Data Visualizations", style={'textAlign': 'center'}),
            dcc.Graph(figure=fig1),  # Sensex Closing Price and Crashes
        ], className="four columns"),

        html.Div([
            dcc.Graph(figure=fig2),  # Market Drawdown
        ], className="four columns"),

        html.Div([
            dcc.Graph(figure=fig3),  # Early Warning Signal
        ], className="four columns")
    ], className="row"),

    html.Br(),
    html.H2("Predict Future Stock Prices", style={'textAlign': 'center'}),
    html.Div([
        dcc.Input(id='days_input', type='number', placeholder='Enter days ahead', min=1, value=30),
        html.Button('Predict', id='predict_button', n_clicks=0),
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),

    html.Br(),
    dcc.Graph(id='prediction_graph')
])

# 5. Callback for prediction
@app.callback(
    Output('prediction_graph', 'figure'),
    Input('predict_button', 'n_clicks'),
    Input('days_input', 'value')
)
def predict_prices(n_clicks, days):
    if n_clicks == 0:
        return {}
    
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date, periods=days+1)[1:]
    future_days_of_year = future_dates.dayofyear
    
    future_preds = model.predict(np.array(future_days_of_year).reshape(-1,1))
    
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds})
    
    fig = px.line(future_df, x='Date', y='Predicted Close', title=f'Predicted Sensex for next {days} days')
    
    return fig

# 6. Run App
if __name__ == '__main__':
    app.run(debug=True)