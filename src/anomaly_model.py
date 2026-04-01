import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go


def load_data(file_path='GOOG_historical.csv'):
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df = df[['Close']].dropna()
    return df


def create_sequences(X, y, timesteps=30):
    X_out, y_out = [], []
    for i in range(len(X) - timesteps):
        X_out.append(X.iloc[i:(i + timesteps)].values)
        y_out.append(y.iloc[i + timesteps])
    return np.array(X_out), np.array(y_out)


def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, activation='tanh', input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.RepeatVector(input_shape[0]),
        tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_shape[1]))
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001), loss='mse')
    return model


def detect_anomalies(model, scaler, df, timesteps=30, threshold=0.229):
    train_size = int(len(df) * 0.85)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    scaler = StandardScaler()
    scaler.fit(train.values.reshape(-1, 1))
    train_scaled = scaler.transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))

    X_train, y_train = create_sequences(
        pd.Series(train_scaled.flatten()), pd.Series(train_scaled.flatten()))
    model.fit(X_train, y_train, epochs=100, batch_size=32,
              validation_split=0.1, verbose=0)

    X_test, y_test = create_sequences(
        pd.Series(test_scaled.flatten()), pd.Series(test_scaled.flatten()))
    pred = model.predict(X_test, verbose=0)
    mae_loss = np.abs(pred.flatten() - y_test)

    test_df = test.iloc[timesteps:].copy()
    test_df['loss'] = mae_loss
    test_df['threshold'] = threshold
    test_df['anomaly'] = test_df['loss'] > threshold
    return test_df, scaler.inverse_transform(test[['Close']])[timesteps:]


def plot_anomalies(anomaly_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anomaly_df.index,
                  y=anomaly_df['Close'], name='Close Price'))
    anomalies = anomaly_df[anomaly_df['anomaly']]
    fig.add_trace(go.Scatter(x=anomalies.index,
                  y=anomalies['Close'], mode='markers', name='Anomaly', marker=dict(color='red', size=10)))
    fig.update_layout(title='Anomaly Detection in Stock Prices',
                      xaxis_title='Date', yaxis_title='Close Price')
    return fig
