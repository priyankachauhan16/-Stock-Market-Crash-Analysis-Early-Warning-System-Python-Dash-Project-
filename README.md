# -Stock-Market-Crash-Analysis-Early-Warning-System-Python-Dash-Project-

🔍 Overview
This project presents an interactive Stock Market Dashboard built using Python, Dash, and Plotly, focusing on historical Sensex data. The dashboard visualizes key financial metrics such as market crashes, drawdowns, and future price predictions using machine learning, offering a powerful tool for investors and analysts to understand trends and detect risk signals.

🎯 Project Objectives
The main goals of this project are to:

Detect daily market crashes based on significant negative returns.

Visualize market drawdowns over time.

Simulate a 2025 early warning system using volatility and trend indicators.

Build a Random Forest model to predict future Sensex prices.

Provide an interactive dashboard to explore these insights in real-time.

📁 Dataset Description
The project uses a cleaned version of historical Sensex stock market data with the following fields:

Date: Daily trading date

Close: Closing index value of the Sensex

Daily_Return: Percentage daily return

Crash_Daily: Boolean flag for market crash days

Drawdown: % drop from historical maximum

Cumulative_Max: Rolling maximum close value

Additional synthetic 2025 data for early warning simulation

📊 Dashboard Features
📉 Sensex Price & Daily Crashes: Line chart with red markers on crash days (e.g. -5% or worse)

📉 Market Drawdown Visualization: Interactive time series with dashed red threshold (-20%)

🚨 Early Warning Signal Simulation (2025): Synthetic dataset with dynamic warnings based on rolling volatility and mean return

🤖 Predict Future Prices: Random Forest regression model predicting next N days of Sensex prices

📱 Interactive UI with Dash:

Predictive input box for user-specified days

Click-to-predict button

Fully scrollable and zoomable graphs with Plotly

🧠 Tools & Techniques Used
Python Libraries: Pandas, NumPy, Scikit-learn, Dash, Plotly

Machine Learning: RandomForestRegressor for time-series modeling

Data Visualization: Interactive plots with Plotly Express and Plotly Graph Objects

Web Dashboard: Built with Dash by Plotly

📌 Key Insights
Major daily crashes were accurately flagged using a -5% threshold on daily returns.

The drawdown plot highlights periods of long-term downturn exceeding 20%.

The early warning system identifies periods of elevated volatility and declining trend.

The predictive model shows the capability to estimate future index values based on seasonality.

🚀 Future Improvements
Use more sophisticated features for price prediction (e.g., lag features, moving averages)

Add sector-wise analysis and heatmaps for stock clusters

Integrate news sentiment analysis with price drops

Deploy as a cloud app using Heroku or Streamlit Cloud

Enable email or SMS alerts based on early warning triggers

🖼️ Project Screenshot

Screenshort(38).png

Screenshot(39).png

💾 How to Run This Project

On terminal write python Stock-Analysis.py

📬 Contact

Priyanka Chauhan

Github Link: https://github.com/priyankachauhan16
