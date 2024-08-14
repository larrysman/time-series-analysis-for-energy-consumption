# import the necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import time
from datetime import datetime
import os
import pickle

# Load the paths to the project folders
main = 'C:\\Users\\Olanrewaju Adegoke\\Desktop\\TechTern\\mywork\\Time_Series_Analysis_Projects\\Energy_Consumption_Forecast_Project'
data = '../data'
model = '../models'
notebk = '../notebooks'
output = '../outputs'
resource = '../resources'
visual = '../visuals'

# Load all the saved models for engineering the features into the future and the total energy model
def load_all_saved_model_and_artifacts():
    os.chdir(model)

    with open('total_energy_model.pkl', 'rb') as file:
        energy_model = pickle.load(file)

    with open('engineered_feature_forecast_model.pkl', 'rb') as file:
        features_forecast = pickle.load(file)

    return energy_model, features_forecast

# instantiate the instances of the loaded model
energy_model, features_forecast = load_all_saved_model_and_artifacts()
os.chdir(output)
total_energy_model = energy_model['total_energy_forecast_model']
hourly_lag = features_forecast['1h_lag']
eight_hour_lag = features_forecast['8h_lag']
twenty_four_hour_lag = features_forecast['24h_lag']
six_day_lag = features_forecast['6d_lag']
mv_avg_2hr_mean = features_forecast['mv_avg_2hr_mean']
mv_avg_2hr_std = features_forecast['mv_avg_2hr_std']
mv_avg_24hr_mean = features_forecast['mv_avg_24hr_mean']
mv_avg_24hr_std = features_forecast['mv_avg_24hr_std']

# Build the streamlit function to predict the total energy consumption
TITLE = 'Energy Consumption Model for the next 1000 hours or more by Olanrewaju Adegoke'
DESCRIPTION = 'You want to estimate the energy consumption for the next 1000 hours, provide the two information and have your energy consumption estimated.'

def energy_consumption_prediction():
    st.title(TITLE)
    st.markdown(DESCRIPTION)

    # Define all the inputs columns parameters
    start_date = st.text_input('Enter the start_datetime (YYYY-MM-DD HH:MM:SS): ')
    num_of_hours = st.number_input('Enter the number of hours: ')   # , format='%d'

    predict_button = st.button('Compute Energy Consumption')
    if predict_button:

        def future_total_energy_prediction(start_date, num_of_hours):
            cols = ['1h_lag', '8h_lag', '24h_lag', '6d_lag', 'mv_avg_2hr_mean',
       'mv_avg_2hr_std', 'mv_avg_24hr_mean', 'mv_avg_24hr_std']

            start_date = pd.to_datetime(start_date)
            date_time_sequence = pd.date_range(start_date, periods=num_of_hours, freq='h')
            date_time_df = pd.DataFrame(date_time_sequence, columns=['date_time'])
            date_time = date_time_df.set_index('date_time')
            date_time['hour'] = date_time.index.hour
            date_time['day'] = date_time.index.dayofweek
            date_time['month'] = date_time.index.month
            date_time['year'] = date_time.index.year
    
            for col in cols:
                
                if col == '1h_lag':
                    pred_1hr_lag = hourly_lag.predict(date_time)
                    pred_1hr_lag_df = pd.DataFrame(pred_1hr_lag, columns=[col], index=date_time.index)
                    
                if col == '8h_lag':
                    
                    pred_8hr_lag = eight_hour_lag.predict(date_time)
                    pred_8hr_lag_df = pd.DataFrame(pred_8hr_lag, columns=[col], index=date_time.index)

                if col == '24h_lag':
                    
                    pred_24hr_lag = twenty_four_hour_lag.predict(date_time)
                    pred_24hr_lag_df = pd.DataFrame(pred_24hr_lag, columns=[col], index=date_time.index)

                if col == '6d_lag':
                    
                    pred_6d_lag = six_day_lag.predict(date_time)
                    pred_6d_lag_df = pd.DataFrame(pred_6d_lag, columns=[col], index=date_time.index)

                if col == 'mv_avg_2hr_mean':
                    
                    pred_mv_avg_2hr_mean = mv_avg_2hr_mean.predict(date_time)
                    pred_mv_avg_2hr_mean_df = pd.DataFrame(pred_mv_avg_2hr_mean, columns=[col], index=date_time.index)

                if col == 'mv_avg_2hr_std':
                    
                    pred_mv_avg_2hr_std = mv_avg_2hr_std.predict(date_time)
                    pred_mv_avg_2hr_std_df = pd.DataFrame(pred_mv_avg_2hr_std, columns=[col], index=date_time.index)

                if col == 'mv_avg_24hr_mean':
                    
                    pred_mv_avg_24hr_mean = mv_avg_24hr_mean.predict(date_time)
                    pred_mv_avg_24hr_mean_df = pd.DataFrame(pred_mv_avg_24hr_mean, columns=[col], index=date_time.index)

                if col == 'mv_avg_24hr_std':

                    pred_mv_avg_24hr_std = mv_avg_24hr_std.predict(date_time)
                    pred_mv_avg_24hr_std_df = pd.DataFrame(pred_mv_avg_24hr_std, columns=[col], index=date_time.index)
            
            futuristic_features = pd.concat([pred_1hr_lag_df, pred_8hr_lag_df, pred_24hr_lag_df, pred_6d_lag_df, pred_mv_avg_2hr_mean_df,
                                    pred_mv_avg_2hr_std_df, pred_mv_avg_24hr_mean_df, pred_mv_avg_24hr_std_df, date_time], axis=1)

            total_energy_predicted = total_energy_model.predict(futuristic_features)
            total_energy_df = pd.DataFrame(total_energy_predicted, columns=[f'future_{num_of_hours}_hours_of_total_energy'], index=date_time.index)
            
            return total_energy_df
        energy_forecast = future_total_energy_prediction(start_date, num_of_hours)
        st.dataframe(energy_forecast)
