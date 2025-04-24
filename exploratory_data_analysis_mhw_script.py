import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import marineHeatWaves as mhw
from collections import defaultdict


# Load the SST dataset on training period
def load_and_prepare_data(file_path, start_date='1982-01-01', end_date='2016-12-31'):

    ds = xr.open_dataset(file_path)
    ds = ds.sel(time=slice(start_date, end_date))
    time = ds['time'].values
    sst = ds['sst'].values
    t = np.array([date.fromisoformat(str(d)[:10]).toordinal() for d in time])
    return ds, sst, t


# Detecting MHWs based on Hobday Definition
def detect_mhw(t, sst):

    mhws, clim = mhw.detect(t, sst)
    return mhws, clim


# Labelling on Binary Events
def label_mhw_events(t, mhws):

    mhw_labels = np.zeros(len(t))
    for i in range(mhws['n_events']):
        start_idx = mhws['index_start'][i]
        end_idx = mhws['index_end'][i] + 1
        mhw_labels[start_idx:end_idx] = 1
    return mhw_labels


# Update by adding variables for MHW labels and climatology
def update_dataset(ds, mhw_labels, clim):

    ds = ds.assign(
        mhw_label=("time", mhw_labels),
        clim_threshold=("time", clim['thresh']),
        clim_seas=("time", clim['seas'])
    )
    return ds


# Summary on MHW Events
def print_mhw_summary(mhws):

    print("Total Marine Heatwave Events:", mhws['n_events'])
    print("\nDetails of Detected Marine Heatwaves:")
    for i in range(mhws['n_events']):
        print(f"Event {i+1}:")
        print(f"  Start Date       : {mhws['date_start'][i]}")
        print(f"  End Date         : {mhws['date_end'][i]}")
        print(f"  Duration         : {mhws['duration'][i]} days")
        print(f"  Max Intensity    : {mhws['intensity_max'][i]:.2f} °C")
        print(f"  Mean Intensity   : {mhws['intensity_mean'][i]:.2f} °C")
        print(f"  Cumulative Int.  : {mhws['intensity_cumulative'][i]:.2f} °C·days")
        print("-" * 40)


# Plots of SST related

def plot_sst_with_mhws(t, sst, mhws, clim, title="SST Time Series with Detected Marine Heatwaves"):

    plot_dates = [date.fromordinal(int(d)) for d in t]

    plt.figure(figsize=(15, 5))
    plt.plot(plot_dates, sst, label='SST', color='steelblue')
    plt.plot(plot_dates, clim['thresh'], label='Climatological Threshold', color='darkorange', linestyle='--')
    plt.plot(plot_dates, clim['seas'], label='Seasonal Climatology', color='green', linestyle=':')

    for i in range(mhws['n_events']):
        start = mhws['date_start'][i]
        end = mhws['date_end'][i]
        plt.axvspan(start, end, color='orange', alpha=0.3)

    plt.xlabel("Time")
    plt.ylabel("SST (°C)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_filled_mhw_areas(t, sst, mhws, clim, start_year, end_year):

    start_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)
    start_ord, end_ord = start_date.toordinal(), end_date.toordinal()

    mask = (t >= start_ord) & (t <= end_ord)
    t_filtered = t[mask]
    sst_filtered = sst[mask]
    clim_thresh_filtered = clim['thresh'][mask]
    clim_seas_filtered = clim['seas'][mask]
    plot_dates = [date.fromordinal(int(d)) for d in t_filtered]

    plt.figure(figsize=(15, 6))
    plt.plot(plot_dates, sst_filtered, label='SST', color='blue', linewidth=1.5)
    plt.plot(plot_dates, clim_thresh_filtered, label='Climatological Threshold', color='red', linestyle='--', linewidth=2)
    plt.plot(plot_dates, clim_seas_filtered, label='Climatology', color='green', linestyle=':', linewidth=2)

    for i in range(mhws['n_events']):
        start = mhws['date_start'][i]
        end = mhws['date_end'][i]
        if start >= start_date and end <= end_date:
            plt.axvspan(start, end, color='orange', alpha=0.4, label='MHW' if i == 0 else None)

    plt.xlabel('Time')
    plt.ylabel('SST (°C)')
    plt.title(f'Marine Heatwaves from {start_year} to {end_year}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Testing Dataset
def prepare_testing_dataset(file_path, start_date='2017-01-01', end_date='2020-12-31'):

    ds = xr.open_dataset(file_path)
    ds = ds.sel(time=slice(start_date, end_date))
    time = ds['time'].values
    sst = ds['sst'].values
    t = np.array([date.fromisoformat(str(d)[:10]).toordinal() for d in time])
    return ds, sst, t


# Labeling test set
def label_mhw_events_for_testing(t, mhws):

    mhw_labels = np.zeros(len(t))
    for i in range(mhws['n_events']):
        start_idx = mhws['index_start'][i]
        end_idx = mhws['index_end'][i] + 1
        mhw_labels[start_idx:end_idx] = 1
    return mhw_labels


# Update the testing dataset by adding MHW labels and climatology data
def update_testing_dataset(ds, mhw_labels, clim):

    ds = ds.assign(
        mhw_label=("time", mhw_labels),
        clim_threshold=("time", clim['thresh']),
        clim_seas=("time", clim['seas'])
    )
    return ds


# Detect MHWs using training period climatology thresholding
def detect_mhw_using_training_climatology(testing_t, testing_sst, training_t, training_sst):

    alternate_climatology = [training_t, training_sst]
    training_years = [date.fromordinal(tt).year for tt in training_t]
    climatologyPeriod = [min(training_years), max(training_years)]
    mhws, clim = mhw.detect(testing_t, testing_sst, climatologyPeriod=climatologyPeriod, alternateClimatology=alternate_climatology)
    return mhws, clim


# Catergories of MHWs
def label_mhw_categories(t, mhws):

    cat_labels = np.zeros(len(t))
    category_mapping = {'Moderate': 1, 'Strong': 2, 'Severe': 3, 'Extreme': 4}
    
    for i in range(mhws['n_events']):
        start_idx = mhws['index_start'][i]
        end_idx = mhws['index_end'][i] + 1  
        cat_value = category_mapping.get(mhws['category'][i], 1)
        cat_labels[start_idx:end_idx] = cat_value
    return cat_labels


# Plot catergories
def plot_mhw_categories(t, cat_labels, title="Marine Heatwave Categories Over Time"):

    plot_dates = [date.fromordinal(int(d)) for d in t]
    
    plt.figure(figsize=(15, 5))
    plt.step(plot_dates, cat_labels, where='post', label='MHW Category', color='purple')
    
    plt.xlabel("Time")
    plt.ylabel("MHW Category")
    plt.title(title)
    plt.yticks([0, 1, 2, 3, 4], ['No MHW', 'Moderate', 'Strong', 'Severe', 'Extreme'])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_mhw_category_counts(mhws):
    
    cat_counts = defaultdict(int) 
    cat_days = defaultdict(int) 

    for i, cat_str in enumerate(mhws['category']):
        cat_counts[cat_str] += 1
        cat_days[cat_str] += mhws['duration'][i]
    print("Number of MHW events (and total days) by category:")
    categories = ["Moderate", "Strong", "Severe", "Extreme"]
    for cat in categories:
        print(f"  {cat}: {cat_counts.get(cat, 0)} events, total days = {cat_days.get(cat, 0)}")
