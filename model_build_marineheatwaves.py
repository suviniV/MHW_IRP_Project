## Importing Libraries
import warnings
warnings.filterwarnings("ignore")
import os
import random
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 


def set_seeds(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def load_data(file_path):

    ds = xr.open_dataset(file_path)
    return ds


# Splitting the dataset into training, validation, and testing sets
def prepare_data_for_modeling(ds):

    sst = ds['sst'].values
    time_index = pd.to_datetime(ds['time'].values)
    month = time_index.month.values.reshape(-1, 1)
    
    train_mask = (time_index >= "1982-01-01") & (time_index <= "2011-12-31")
    valid_mask = (time_index >= "2012-01-01") & (time_index <= "2016-12-31")
    test_mask  = (time_index >= "2017-01-01") & (time_index <= "2020-12-31")
    
    train_sst = sst[train_mask]
    valid_sst = sst[valid_mask]
    test_sst  = sst[test_mask]
    
    train_month = month[train_mask]
    valid_month = month[valid_mask]
    test_month  = month[test_mask]
    
    train_time = time_index[train_mask]
    valid_time = time_index[valid_mask]
    test_time  = time_index[test_mask]
    
    return (train_sst, train_month, train_time), (valid_sst, valid_month, valid_time), (test_sst, test_month, test_time)


# Create lagged input-output sequences
def create_sequences(data, n_lag, n_out):
    
    X, y = [], []
    for i in range(len(data) - n_lag - n_out + 1):
        seq_x = data[i:i+n_lag, :]       
        seq_y = data[i+n_lag:i+n_lag+n_out, 0]  
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


# Normalize input data and create lagged sequences
def scale_and_create_sequences(train_sst, train_month, valid_sst, valid_month, test_sst, test_month, n_lag=7, n_out=7):

    train_data = np.hstack([train_sst.reshape(-1, 1), train_month])
    valid_data = np.hstack([valid_sst.reshape(-1, 1), valid_month])
    test_data  = np.hstack([test_sst.reshape(-1, 1), test_month])
    
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    valid_scaled = scaler.transform(valid_data)
    test_scaled  = scaler.transform(test_data)
    
    X_train, y_train = create_sequences(train_scaled, n_lag, n_out)
    X_valid, y_valid = create_sequences(valid_scaled, n_lag, n_out)
    X_test, y_test   = create_sequences(test_scaled, n_lag, n_out)
    
    sst_min = scaler.data_min_[0]
    sst_max = scaler.data_max_[0]
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, sst_min, sst_max

# LSTM model Build
def build_lstm_model(n_lag, n_features, n_seq):

    model = Sequential()
    model.add(LSTM(64, input_shape=(n_lag, n_features)))
    model.add(Dense(n_seq, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    return model

# Train LSTM model with training and validation data.
def fit_lstm_model(model, X_train, y_train, X_valid, y_valid, epochs, batch_size):

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_valid, y_valid),
        shuffle=False
    )
    return history

# CNN 1D model Build
def build_cnn_model(n_lag, n_features, n_seq):
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_lag, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(n_seq, activation='linear'))
    model.compile(optimizer='adam', loss='mae')
    return model


# Train CNN 1D model with training and validation data.
def fit_cnn_model(model, X_train, y_train, X_valid, y_valid, epochs, batch_size):

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_valid, y_valid),
        shuffle=False
    )
    return history


# Forecasts and invert the scaling for SST
def forecast_and_invert(model, X, y, sst_min, sst_max):

    yhat = model.predict(X)
    yhat_inverted = yhat * (sst_max - sst_min) + sst_min
    y_inverted    = y * (sst_max - sst_min) + sst_min
    return yhat_inverted, y_inverted


# Functions for Evaluations

def evaluate_forecasts(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def evaluate_forecasts_all(y_true, y_pred):

    n_seq = y_true.shape[1]
    rmse_list = []
    for i in range(n_seq):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        print(f"Forecast horizon t+{i+1}: RMSE = {rmse:.4f}")
        rmse_list.append(rmse)
    return rmse_list


def plot_training_history(history):

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_actual_vs_predicted(time, actual, predicted, title="Actual vs Predicted SST"):

    plt.figure(figsize=(12, 6))
    plt.plot(time, actual, label="Actual SST", color="blue")
    plt.plot(time, predicted, label="Predicted SST", color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("SST (°C)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_evaluation_metrics_all(y_true, y_pred):
 
    n_seq = y_true.shape[1]
    horizons = np.arange(1, n_seq + 1)
    rmse_list, mae_list, r2_list = [], [], []

    for i in range(n_seq):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        print(f"Forecast horizon t+{i+1}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R2 = {r2:.4f}")

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    axs[0].plot(horizons, rmse_list, marker='o')
    axs[0].set_title("RMSE vs Forecast Horizon")
    axs[0].set_xlabel("Forecast Horizon (t+X)")
    axs[0].set_ylabel("RMSE")
    axs[0].grid(True)

    axs[1].plot(horizons, mae_list, marker='o', color='orange')
    axs[1].set_title("MAE vs Forecast Horizon")
    axs[1].set_xlabel("Forecast Horizon (t+X)")
    axs[1].set_ylabel("MAE")
    axs[1].grid(True)

    axs[2].plot(horizons, r2_list, marker='o', color='green')
    axs[2].set_title("R2 Score vs Forecast Horizon")
    axs[2].set_xlabel("Forecast Horizon")
    axs[2].set_ylabel("R2 Score")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    metrics_df = pd.DataFrame({
        "Forecast Horizon": horizons,
        "RMSE": rmse_list,
        "MAE": mae_list,
        "R2": r2_list
    })
    return metrics_df


def evaluate_mhw_metrics(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    fdr = FP / (FP + TP) if (FP + TP) > 0 else 0.0
    
    return {
        "Precision": precision,
        "Recall (TPR)": recall,
        "F1 Score": f1,
        "Accuracy": accuracy,
        "FPR": fpr,
        "TPR": recall, 
        "FNR": fnr,
        "FDR": fdr
    }

def align_datasets(ds_actual, ds_predicted):

    common_times = np.intersect1d(ds_actual['time'].values, ds_predicted['time'].values)
    aligned_actual = ds_actual.sel(time=common_times)
    aligned_predicted = ds_predicted.sel(time=common_times)
    return aligned_actual, aligned_predicted


def evaluate_mhw_forecast_horizons(adjusted_test_time, y_test_inverted, yhat_test_inverted, train_t, train_sst, eda):

    n_seq = y_test_inverted.shape[1]
    metrics_dict = {}
    
    for i in range(n_seq):
        effective_time = np.array([t + timedelta(days=i) for t in adjusted_test_time])
        
        actual_ds = xr.Dataset(
            {"sst": (("time",), y_test_inverted[:, i])},
            coords={"time": effective_time}
        )
        predicted_ds = xr.Dataset(
            {"sst": (("time",), yhat_test_inverted[:, i])},
            coords={"time": effective_time}
        )
        
        effective_time_ord = np.array([d.toordinal() for d in effective_time])
        
        # Detect and label MHW events
        mhws_actual, clim_actual = eda.detect_mhw_using_training_climatology(effective_time_ord, y_test_inverted[:, i], train_t, train_sst)
        actual_labels = eda.label_mhw_events_for_testing(effective_time_ord, mhws_actual)
        updated_actual_ds = eda.update_testing_dataset(actual_ds, actual_labels, clim_actual)
        
        mhws_pred, clim_pred = eda.detect_mhw_using_training_climatology(effective_time_ord, yhat_test_inverted[:, i], train_t, train_sst)
        pred_labels = eda.label_mhw_events_for_testing(effective_time_ord, mhws_pred)
        updated_predicted_ds = eda.update_testing_dataset(predicted_ds, pred_labels, clim_pred)
        
        aligned_actual, aligned_predicted = align_datasets(updated_actual_ds, updated_predicted_ds)
        actual_bin = aligned_actual['mhw_label'].values
        pred_bin = aligned_predicted['mhw_label'].values
        
        horizon_metrics = evaluate_mhw_metrics(actual_bin, pred_bin)
        metrics_dict[f"t+{i+1}"] = horizon_metrics
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(actual_bin, pred_bin)
        print(f"Confusion Matrix for forecast horizon t+{i+1}:")
        print(cm)
        
        plot_confusion_matrix(cm, classes=['No MHW', 'MHW'], title=f'Confusion Matrix for t+{i+1}')
    
    return metrics_dict


def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def metrics_to_dataframe(metrics_dict):

    rows = []
    for horizon, metrics in metrics_dict.items():
        row = {"Horizon": horizon}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.set_index("Horizon", inplace=True)
    return df


# Plotting Functions 

def plot_sst_scatter_forecasts(y_true, y_pred, title="Actual vs Predicted SST for Forecast Horizons"):

    n_seq = y_true.shape[1]
    fig, axs = plt.subplots(2, (n_seq + 1) // 2, figsize=(16, 8))
    axs = axs.flatten()

    for i in range(n_seq):
        axs[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
        axs[i].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  
        axs[i].set_title(f"Forecast t+{i+1}")
        axs[i].set_xlabel("Actual SST")
        axs[i].set_ylabel("Predicted SST")
        axs[i].grid(True)

    for j in range(n_seq, len(axs)):
        fig.delaxes(axs[j]) 

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_metric_by_region(metric_results, metric_name="RMSE", ylabel="°C"):

    plt.figure(figsize=(8, 6))
    horizons = list(range(1, len(next(iter(metric_results.values()))) + 1))

    for region, values in metric_results.items():
        plt.plot(horizons, values, marker='o', label=region)

    plt.title(f"{metric_name} vs Forecast Horizon")
    plt.xlabel("Forecast Lead [days]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predicted_vs_climatology(time_array, predicted_sst, clim_ds, region_name=""):

    clim = clim_ds['seas']  
    thresh = clim_ds['thresh']

    plt.figure(figsize=(14, 6))
    plt.plot(time_array, predicted_sst, label="Predicted SST", color="red", linewidth=2)
    plt.plot(time_array, thresh, label="Threshold (90th Percentile)", color="gray", linestyle=":", linewidth=2)

    mhw_mask = predicted_sst > thresh
    plt.fill_between(time_array, thresh, predicted_sst, where=mhw_mask, color='orange', alpha=0.3, label="MHW Detected")

    plt.title(f"{region_name} - Forecast Horizon")
    plt.xlabel("Date")
    plt.ylabel("SST (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


