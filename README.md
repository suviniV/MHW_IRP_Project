# MHW_IRP_Project

**Prediction of Marine Heatwaves and Their Impact on Ocean Productivity using Deep Learning Approach**

This project investigates the identification and forecasting of Marine Heatwaves (MHWs) and evaluates their impact on ocean productivity, using satellite-derived Sea Surface Temperature (SST) and Chlorophyll-a (Chl-a) datasets. Deep learning models such as LSTM and CNN are utilized to perform time-series forecasting of MHWs and Chl-a. The Regions covered here are Bay of Bengal, Gulf of Mannar, Laccadive Sea and Sri Lankan Coast


## Project Structure

```bash
MHW_IRP_PROJECT/
│
├── Data_chl_2020/                     # Chlorophyll-a (Chl-a) Region Wise Area Averaged Datasets Used 
├── Data_sst_2020/                     # Sea Surface Temperature (SST) Region Wise Datasets Used
│
├── exploratory_data_analysis.ipynb               # EDA for SST & Chl-a Data on the Regions with Visualizations
├── exploratory_data_analysis_script.py           # Script consisting of functions related to EDA Analysis
├── exploratory_data_analysis_mhw.ipynb           # EDA focusing on MHW detection 
├── exploratory_data_analysis_mhw_script.py       # Script consisting of functions MHW EDA
│
├── marineHeatWaves.py                             # MHW detection utilities and labeling (based on Hobday et al., 2016)
│
├── chl_model_CNN.ipynb                            # CNN-based Chl-a prediction 
├── chl_model_LSTM.ipynb                           # LSTM-based Chl-a prediction 
├── mhw_model_CNN.ipynb                            # CNN-based SST prediction for MHWs
├── mhw_model_LSTM.ipynb                           # LSTM-based SST prediction for MHWs
│
├── model_build_chlorophyll.py                     # Script to build functions related to Chl-a forecasting models
├── model_build_marineheatwaves.py                 # Script to build functions related to MHW forecasting models
│
├── mhw_chl_impact.py                              # Analyzing the impact of MHWs on Chl-a Preprocessing Script
├── mhw_impact_on_chl_EDA.ipynb                    # EDA on MHW impact on Chl-a