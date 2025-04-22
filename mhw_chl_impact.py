import xarray as xr
import pandas as pd
import numpy as np
from scipy.signal import detrend
import matplotlib.pyplot as plt
from exploratory_data_analysis_mhw_script import detect_mhw, label_mhw_events
from model_build_chlorophyll import load_and_merge_data


### Removing the climatalogy and detrending
def compute_anomaly_detrended(ds, var):
    day_of_year = ds.time.dt.dayofyear
    clim = ds[var].groupby(day_of_year).mean('time')
    anomaly = ds[var].groupby('time.dayofyear') - clim
    return detrend(anomaly.values)


def get_mhw_events(time, sst):
    t_ord = np.array([d.toordinal() for d in time])
    mhws, clim = detect_mhw(t_ord, sst)
    labels = label_mhw_events(t_ord, mhws)
    return mhws, labels


# Plot of CHL-a anomaies with SST Anomalies
def plot_mhw_impact_profile(ds, mhw_labels, max_events=50, days_before=200, days_after=200):
    df = pd.DataFrame({
        "time": ds.time.values,
        "sst": ds["sst_anom_detrended"].values,
        "chl": ds["chl_anom_detrended"].values,
        "mhw_label": mhw_labels
    })
    df['time'] = pd.to_datetime(df['time'])
    df['mhw_shift'] = df['mhw_label'].shift(-1)
    mhw_end_dates = df[(df['mhw_label'] == 1) & (df['mhw_shift'] == 0)]['time'].reset_index(drop=True)

    num_events = min(max_events, len(mhw_end_dates))
    fig, axes = plt.subplots(nrows=num_events, figsize=(14, 5 * num_events), sharex=False)
    if num_events == 1:
        axes = [axes]

    for i, mhw_end in enumerate(mhw_end_dates[:num_events]):
        end_idx = df[df['time'] == mhw_end].index[0]
        start_idx = end_idx
        while start_idx > 0 and df.loc[start_idx - 1, 'mhw_label'] == 1:
            start_idx -= 1
        mhw_start = df.loc[start_idx, 'time']
        duration = (mhw_end - mhw_start).days

        start_date = mhw_start - pd.Timedelta(days=days_before)
        end_date = mhw_end + pd.Timedelta(days=days_after)
        zoom_df = df[(df['time'] >= start_date) & (df['time'] <= end_date)].copy()
        zoom_df['days_from_start'] = (zoom_df['time'] - mhw_start).dt.days

        ax1 = axes[i]
        ax1.plot(zoom_df['days_from_start'], zoom_df['sst'], label="SST Anomaly (°C)", color='blue')
        ax2 = ax1.twinx()
        ax2.plot(zoom_df['days_from_start'], zoom_df['chl'], label="Chl-a Anomaly (mg/m³)", color='green')

        ax1.axvline(0, linestyle='--', color='purple', label="MHW Start")
        ax1.axvline(duration, linestyle='--', color='red', label="MHW End")

        ax1.set_xticks(np.arange(-days_before, days_after + 1, 50))
        ax1.set_xlim(-days_before, days_after)
        ax1.set_xlabel("Days from MHW Start", fontsize=12)

        ax1.set_ylabel("SST Anomaly (°C)", color='blue')
        ax2.set_ylabel("Chl-a Anomaly (mg/m³)", color='green')

        ax1.set_title(f"MHW Event {i+1}: {mhw_start.date()} to {mhw_end.date()} (Duration: {duration} days)")
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_detrended_variable(ds, var, label, units, color='blue'):

    time = pd.to_datetime(ds.time.values)
    values = ds[var].values

    plt.figure(figsize=(14, 5))
    plt.plot(time, values, label=label, color=color)
    plt.axhline(0, linestyle='--', color='black', linewidth=0.8)
    plt.title(f"{label} Over Time", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel(f"{label} [{units}]", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()



## Full process upto plotting
def run_mhw_anomaly_analysis(chl_file_path, sst_file_path, max_events=50, days_before=200, days_after=200):
    merged_ds = load_and_merge_data(chl_file_path, sst_file_path)

    sst_anom_detrended = compute_anomaly_detrended(merged_ds, "sst")
    chl_anom_detrended = compute_anomaly_detrended(merged_ds, "chl")

    merged_ds["sst_anom_detrended"] = (["time"], sst_anom_detrended)
    merged_ds["chl_anom_detrended"] = (["time"], chl_anom_detrended)

    time = pd.to_datetime(merged_ds.time.values)
    sst = merged_ds["sst"].values
    mhws, labels = get_mhw_events(time, sst)

    plot_mhw_impact_profile(merged_ds, labels, max_events=max_events, days_before=days_before, days_after=days_after)

    return merged_ds, mhws, labels
