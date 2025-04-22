import warnings
warnings.filterwarnings("ignore")
import os
import random
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


def set_seeds(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


# Merge SST and Chl-a Data
def load_and_merge_data(chl_file, sst_file):
    ds_chl = xr.open_dataset(chl_file)
    ds_sst = xr.open_dataset(sst_file)
    
    chl_time_range = ds_chl.time
    sst_ds = ds_sst.sel(time=chl_time_range)    
    sst_ds, ds_chl_aligned = xr.align(sst_ds, ds_chl, join="inner")
    merged_ds = xr.merge([sst_ds, ds_chl_aligned])

    return merged_ds


# Splitting the dataset into training, validation, and testing sets
def prepare_data_for_modeling(merged_ds):

    sst = merged_ds['sst'].values
    chl = merged_ds['chl'].values
    time_index = pd.to_datetime(merged_ds.time.values)
    month = time_index.month.values.reshape(-1, 1)
    
    train_mask = (time_index >= "1993-01-01") & (time_index <= "2011-12-31")
    valid_mask = (time_index >= "2012-01-01") & (time_index <= "2016-12-31")
    test_mask  = (time_index >= "2017-01-01") & (time_index <= "2020-12-31")
    
    train_sst = sst[train_mask]
    valid_sst = sst[valid_mask]
    test_sst  = sst[test_mask]
    
    train_chl = chl[train_mask]
    valid_chl = chl[valid_mask]
    test_chl  = chl[test_mask]
    
    train_month = month[train_mask]
    valid_month = month[valid_mask]
    test_month  = month[test_mask]
    
    train_time = time_index[train_mask]
    valid_time = time_index[valid_mask]
    test_time  = time_index[test_mask]
    
    return (train_sst, train_chl, train_month, train_time), (valid_sst, valid_chl, valid_month, valid_time), (test_sst, test_chl, test_month, test_time)


# Create lagged input-output sequences
def create_sequences(data, n_lag, n_out):
    
    X, y = [], []
    for i in range(len(data) - n_lag - n_out + 1):
        seq_x = data[i:i+n_lag, :2]   
        seq_y = data[i+n_lag:i+n_lag+n_out, 2] 
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# Normalize input data and create lagged sequences
def scale_and_create_sequences(train_sst, train_chl, train_month,
                               valid_sst, valid_chl, valid_month,
                               test_sst, test_chl, test_month, n_lag, n_out):

    train_data = np.hstack([train_sst.reshape(-1, 1), train_month, train_chl.reshape(-1, 1)])
    valid_data = np.hstack([valid_sst.reshape(-1, 1), valid_month, valid_chl.reshape(-1, 1)])
    test_data  = np.hstack([test_sst.reshape(-1, 1), test_month, test_chl.reshape(-1, 1)])
    
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    valid_scaled = scaler.transform(valid_data)
    test_scaled  = scaler.transform(test_data)
    
    X_train, y_train = create_sequences(train_scaled, n_lag, n_out)
    X_valid, y_valid = create_sequences(valid_scaled, n_lag, n_out)
    X_test, y_test   = create_sequences(test_scaled, n_lag, n_out)
    
    chl_min = scaler.data_min_[2]
    chl_max = scaler.data_max_[2]
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler, chl_min, chl_max


# LSTM model Build
def build_lstm_model(n_lag, n_features, n_out):
    model = Sequential()
    model.add(LSTM(50, input_shape=(n_lag, n_features), return_sequences=False)),
    model.add(Dense(n_out, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    return model


# Train LSTM model with training and validation data.
def train_lstm_model(model, X_train, y_train, X_valid, y_valid, epochs, batch_size):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_valid, y_valid),
        shuffle = True
    )
    return history


# CNN 1D model Build
def build_cnn_model(n_lag, n_features, n_out):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_lag, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(n_out, activation='linear'))
    model.compile(optimizer='adam', loss='mae')

    return model


# Train CNN 1D model with training and validation data.
def train_cnn_model(model, X_train, y_train, X_valid, y_valid, epochs, batch_size):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,                
        restore_best_weights=True 
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,  
        batch_size=batch_size,
        validation_data=(X_valid, y_valid),
        callbacks=[early_stopping],
    )
    return history

# Invert MinMaxScaler for Chlorophyll-a values.
def invert_chl_scaling(scaled_chl, chl_min, chl_max):
    return scaled_chl * (chl_max - chl_min) + chl_min


# Evaluate model forecasts 
def evaluate_forecasts(actual, predicted, n_seq):
    rmses = []
    maes = []
    r2_scores = []

    for i in range(n_seq):
        y_true = actual[:, i]
        y_pred = predicted[:, i]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"t+{i+1} RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

        rmses.append(rmse)
        maes.append(mae)
        r2_scores.append(r2)

    return rmses, maes, r2_scores
    
# Plots
def plot_training_history(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.show()


def plot_actual_vs_predicted(y_test_inv, y_pred_inv):
    plt.figure(figsize=(15, 10))
    n_steps = y_test_inv.shape[1]
    for i in range(n_steps):
        plt.subplot(2, 4, i+1)
        plt.scatter(y_test_inv[:, i], y_pred_inv[:, i], alpha=0.5)
        min_val = min(y_test_inv[:, i].min(), y_pred_inv[:, i].min())
        max_val = max(y_test_inv[:, i].max(), y_pred_inv[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.title(f'Forecast t+{i+1}')
        plt.xlabel('Actual Chl-a')
        plt.ylabel('Predicted Chl-a')
    plt.tight_layout()
    plt.suptitle('Actual vs Predicted Chlorophyll-a Concentrations', y=1.02)
    plt.show()


def plot_chl_timeseries(time_test, y_test_inv, y_pred_inv, n_lag, n_out):

    plt.figure(figsize=(16, 10))
    for i in range(y_test_inv.shape[1]):
        plt.subplot(2, 4, i+1)
        # Adjust the time index to align with the forecast horizon
        adjusted_time_index = time_test[n_lag + i: len(time_test) - n_out + i + 1]
        df = pd.DataFrame({
            'Actual': y_test_inv[:, i],
            'Predicted': y_pred_inv[:, i]
        }, index=adjusted_time_index)
        plt.plot(df.index, df['Actual'], label='Actual', color='blue')
        plt.plot(df.index, df['Predicted'], label='Predicted', color='red', linestyle='--')
        plt.title(f'Forecast t+{i+1}')
        plt.xlabel('Time')
        plt.ylabel('Chlorophyll-a Concentration')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
    plt.suptitle('Actual vs Predicted Chlorophyll-a Concentrations Time Series', y=1.02)
    plt.show()

def plot_t1_timeseries(time_test, y_test_inv, y_pred_inv, n_lag, n_out):

    adjusted_time_index = time_test[n_lag: len(time_test) - n_out + 1]
    actual_t1 = y_test_inv[:, 0]
    pred_t1   = y_pred_inv[:, 0]
    
    plt.figure(figsize=(12, 6))
    plt.plot(adjusted_time_index, actual_t1, label='Actual t+1', color='blue', linewidth=2)
    plt.plot(adjusted_time_index, pred_t1, label='Predicted t+1', color='red', linestyle='--', linewidth=2)
    plt.title("t+1 Time Series Forecast: Actual vs. Predicted")
    plt.xlabel("Time")
    plt.ylabel("Chlorophyll-a Concentration")
    plt.legend()
    plt.grid(True)
    plt.show()