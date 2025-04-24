import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import linregress
from statsmodels.tsa.stattools import ccf


# Time Series plot 
def plot_time_series(ds: xr.Dataset, var: str, figsize=(14, 5)):
    unit = '°C' if var.lower() == 'sst' else 'mgm-3'
    plt.figure(figsize=figsize)
    plt.plot(ds.time, ds[var])
    plt.title(f'Time Series of {var.upper()}')
    plt.xlabel('Date')
    plt.ylabel(f'{var.upper()} [{unit}]' if unit else var.upper())
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot linear trend analysis.
def plot_actual_trend(ds: xr.Dataset, var: str, color: str = 'blue', label: str = '', figsize=(14, 5)):

    df = ds[[var]].to_dataframe().dropna().reset_index()
    
    df['year'] = df['time'].dt.year + df['time'].dt.dayofyear / 365.25
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(df['year'], df[var])
    trend_per_decade = slope * 10
    error_per_decade = std_err * 10
    r_squared = r_value ** 2
    unit = '°C' if var.lower() == 'sst' else 'mgm-3'
    
    plt.figure(figsize=figsize)
    plt.plot(df['year'], df[var], label=var.upper(), color=color, linewidth=1)
    plt.plot(df['year'], intercept + slope * df['year'], 'k-', linewidth=1.5,
             label=f'Trend = {trend_per_decade:.2f} ± {error_per_decade:.2f} {unit}/decade\nR² = {r_squared:.2f}')
    
    plt.title(f'{label or var.upper()} Trend with Linear Fit')
    plt.xlabel('Year')
    plt.ylabel(f'{var.upper()} [{unit}]' if unit else var.upper())
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'trend_per_decade': trend_per_decade,
        'r_squared': r_squared
    }


# Compute anomalies based on daily climatology.
def compute_daily_anomalies(ds: xr.Dataset, var: str) -> xr.DataArray:
    day_of_year = ds.time.dt.dayofyear
    clim = ds[var].groupby(day_of_year).mean('time')
    anomalies = ds[var].groupby('time.dayofyear') - clim

    return anomalies


# Anomalies Plot
def plot_anomalies(anomalies: xr.DataArray, var: str, color: str = 'gray', figsize=(14, 5)):

    plt.figure(figsize=figsize)
    plt.plot(anomalies.time, anomalies, color=color)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title(f'Daily Anomalies of {var.upper()}')
    plt.xlabel('Date')
    plt.ylabel(f'{var.upper()} Anomaly')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot of the trend in Anomalies
def plot_trend_on_anomaly(anom: xr.DataArray, var: str, color: str = 'gray', label: str = '', figsize=(14, 5)):

    df = anom.to_dataframe(name='anomaly').dropna().reset_index()
    df['year'] = df['time'].dt.year + df['time'].dt.dayofyear / 365.25
    slope, intercept, r_value, p_value, std_err = linregress(df['year'], df['anomaly'])
    trend_decade = slope * 10
    error_decade = std_err * 10
    r_squared = r_value ** 2
    unit = '°C' if var.lower() == 'sst' else 'mgm-3'

    plt.figure(figsize=figsize)
    plt.plot(df['year'], df['anomaly'], color=color, label=f'{var.upper()} Anomaly')
    plt.plot(df['year'], intercept + slope * df['year'], 'k-', linewidth=1.5,
             label=f'Trend = {trend_decade:.2f} ± {error_decade:.2f} {unit}/decade\nR² = {r_squared:.2f}')
    
    plt.title(f'{label or var.upper()} Anomaly Trend with Linear Fit')
    plt.xlabel('Year')
    plt.ylabel(f'{var.upper()} Anomaly')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'trend_per_decade': trend_decade,
        'r_squared': r_squared
    }


# Plot Climatology
def plot_daily_seasonality(ds: xr.Dataset, var: str, color: str = 'orange', figsize=(12, 5)):

    climatology = ds[var].groupby('time.dayofyear').mean('time')
    plt.figure(figsize=figsize)
    plt.plot(climatology.dayofyear, climatology, color=color)
    plt.title(f'Daily Climatology of {var.upper()} (Seasonal Cycle)')
    plt.xlabel('Day of Year')
    plt.ylabel(f'{var.upper()}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return climatology


# Full data analysis process
def analyze_timeseries(ds: xr.Dataset, var: str, color: str = 'blue', label: str = None):

    if label is None:
        label = var.upper()
    
    print(f"===== Analysis of {label} =====")

    print("\n1. Basic Time Series")
    plot_time_series(ds, var)
    
    print("\n2. Trend Analysis")
    trend_stats = plot_actual_trend(ds, var, color=color, label=label)
    
    print("\n3. Anomaly Calculation")
    anomalies = compute_daily_anomalies(ds, var)
    plot_anomalies(anomalies, var, color=color)
    
    print("\n4. Anomaly Trend Analysis")
    anomaly_trend_stats = plot_trend_on_anomaly(anomalies, var, color=color, label=label)
    
    print("\n5. Seasonal Cycle Analysis")
    seasonality = plot_daily_seasonality(ds, var, color=color)
    
    return {
        'raw_trend': trend_stats,
        'anomaly_trend': anomaly_trend_stats,
        'anomalies': anomalies,
        'seasonality': seasonality
    }


# Analyze cross-correlation between SST and Chl-a
def analyze_sst_chl_correlation(file_path_sst, file_path_chl, max_lag=365, plot_results=True):

    df_sst = xr.open_dataset(file_path_sst)
    df_chl = xr.open_dataset(file_path_chl)

    sst_series = df_sst['sst'].values
    chl_series = df_chl['chl'].values
    time_sst = pd.to_datetime(df_sst['time'].values)
    time_chl = pd.to_datetime(df_chl['time'].values)

    # Obtaining common time range in both dataset Chl-a/SST
    common_time = np.intersect1d(time_sst, time_chl)
    sst_series_pd = pd.Series(sst_series, index=time_sst).loc[common_time]
    chl_series_pd = pd.Series(chl_series, index=time_chl).loc[common_time]

    # Standardize both series
    sst_std = (sst_series_pd - sst_series_pd.mean()) / sst_series_pd.std()
    chl_std = (chl_series_pd - chl_series_pd.mean()) / chl_series_pd.std()

    ccf_vals = ccf(chl_std, sst_std)[:max_lag + 1]
    lags = np.arange(0, max_lag + 1)

    ccf_df = pd.DataFrame({'lag': lags, 'correlation': ccf_vals})
    
    max_corr_idx = np.argmax(ccf_vals)
    max_corr_lag = lags[max_corr_idx]
    max_corr_val = ccf_vals[max_corr_idx]

    first_zero_cross_idx = None
    for i in range(1, len(ccf_vals)):
        if ccf_vals[i - 1] < 0 and ccf_vals[i] >= 0:
            first_zero_cross_idx = i
            break

    first_zero_cross_lag = lags[first_zero_cross_idx] if first_zero_cross_idx is not None else None

    second_zero_cross_idx = None
    for i in range(max_corr_idx + 1, len(ccf_vals)):
        if ccf_vals[i - 1] > 0 and ccf_vals[i] <= 0:
            second_zero_cross_idx = i
            break

    second_zero_cross_lag = lags[second_zero_cross_idx] if second_zero_cross_idx is not None else None

    if plot_results:
        plt.figure(figsize=(12, 5))
        plt.plot(lags, ccf_vals, marker='o', label='(SST → Chl-a)')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel("Lag (days) — Positive lag")
        plt.ylabel("Cross-correlation")
        plt.title("Cross-Correlation between SST and Chlorophyll-a")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(f"First zero-crossing (rises above 0): Lag = {first_zero_cross_lag} days")
    print(f"Maximum correlation: Lag = {max_corr_lag} days, Correlation = {max_corr_val:.4f}")
    print(f"Second zero-crossing (drops below 0 after peak): Lag = {second_zero_cross_lag} days")

    results = {
        'ccf_results': ccf_df,
        'first_zero_crossing': first_zero_cross_lag,
        'max_correlation': {
            'lag': max_corr_lag, 
            'value': max_corr_val
        },
        'second_zero_crossing': second_zero_cross_lag,
        'standardized_series': {
            'sst': sst_std,
            'chl': chl_std,
            'time': common_time
        }
    }
    
    return results


# Correlation of SST and CHL-a
def plot_sst_chl_correlation_scatter(file_path_sst: str, file_path_chl: str,
                                      var_sst: str = 'sst', var_chl: str = 'chl',
                                      label_sst: str = 'SST (°C)', label_chl: str = 'Chl-a (mg m$^{-3}$)',
                                      title: str = 'Correlation of SST and Chlorophyll-a',
                                      figsize: tuple = (8, 6), verbose: bool = True):

    ds_sst = xr.open_dataset(file_path_sst)
    ds_chl = xr.open_dataset(file_path_chl)

    sst_vals = ds_sst[var_sst].values
    chl_vals = ds_chl[var_chl].values
    time_sst = pd.to_datetime(ds_sst['time'].values)
    time_chl = pd.to_datetime(ds_chl['time'].values)

    common_time = np.intersect1d(time_sst, time_chl)
    sst_series = pd.Series(sst_vals, index=time_sst).loc[common_time]
    chl_series = pd.Series(chl_vals, index=time_chl).loc[common_time]
    valid_idx = sst_series.dropna().index.intersection(chl_series.dropna().index)
    sst_series = sst_series.loc[valid_idx]
    chl_series = chl_series.loc[valid_idx]

    slope, intercept, r_value, p_value, _ = linregress(sst_series, chl_series)

    plt.figure(figsize=figsize)
    plt.scatter(sst_series, chl_series, alpha=0.5, edgecolor='k', label='Data points')

    x_fit = np.linspace(sst_series.min(), sst_series.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, 'r-', lw=2,
             label=f'Linear Fit:\n$y={slope:.3f}x$\n$r={r_value:.3f}$')

    plt.title(title)
    plt.xlabel(label_sst)
    plt.ylabel(label_chl)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if verbose:
        print("=== SST & Chl-a Correlation Metrics ===")
        print(f"Slope:     {slope:.4f}")
        print(f"r-value:   {r_value:.4f} (Correlation)")
        print(f"p-value:   {p_value:.4e}")

    return {
        'slope': slope,
        'r_value': r_value,
        'p_value': p_value
    }
