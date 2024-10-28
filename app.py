import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the model and scaler
with open("C:/Users/gourav/Desktop/PROGRAMMING LANGUAGES/stock_prediction_model/stock_model.sav", "rb") as model_file:
    model = pickle.load(model_file)
    
with open("C:/Users/gourav/Desktop/PROGRAMMING LANGUAGES/stock_prediction_model/scaler.sav", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

filepath = "C:/Users/gourav/Desktop/PROGRAMMING LANGUAGES/stock_prediction_model/dataset/NvidiaStock.csv"
df = pd.read_csv(filepath,parse_dates=['date'])
dfShow = pd.read_csv(filepath)
dfShow = dfShow.drop(df.columns[0], axis=1)
df = df.reset_index()['close']
data_array = scaler.fit_transform(np.array(df).reshape(-1,1))
#df_scaled = scaler.transform(df)
#data_array = df_scaled

# Title of the app
st.title("Stock Prediction Using LSTM Model")

# Display a sample of the dataset
st.subheader("Sample of Training Data")
st.write(dfShow.head())  


st.subheader("Next 30-Day Stock Price Prediction")

start_prediction = st.button("Predict Next 30 Days")

if start_prediction:
    x_input = data_array[-100:].reshape(1,-1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    out = []  
    i = 0
    
    while i < 30:
        if len(temp_input) > 100:
            x_input = np.array(temp_input[1:]).reshape(1,-1).reshape((1, 100, 1))
            y_pred = model.predict(x_input, verbose=0)
            temp_input.extend(y_pred[0].tolist())
            temp_input = temp_input[1:]
            out.extend(y_pred.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, 100, 1))
            y_pred = model.predict(x_input, verbose=0)
            temp_input.extend(y_pred[0].tolist())
            out.extend(y_pred.tolist())
            i=i+1


    out = scaler.inverse_transform(np.array(out).reshape(-1, 1))

    # Plotting
    fig, ax = plt.subplots()
    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 131)
    
    # Plot past and predicted values
    ax.plot(day_new, scaler.inverse_transform(data_array[-100:]), label="Last 100 days")
    ax.plot(day_pred, out, label="Next 30 days prediction")
    ax.legend()
    
    # Display plot
    st.pyplot(fig)

    # Display prediction values
    predicted_values_df = pd.DataFrame(out.flatten()).T  

    st.write("Predicted values for the next 30 days:")
    st.table(predicted_values_df)
