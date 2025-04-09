from prophet import Prophet
import pandas as pd

def train_model(df):
    df_prophet = df[['date', 'fire_count']].rename(columns={'date': 'ds', 'fire_count': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return model, forecast
