import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

with open("stock_model.sav", "rb") as model_file:
    model = pickle.load(model_file)
    
with open("scaler.sav", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

filepath = "dataset/NvidiaStock.csv"
df = pd.read_csv(filepath,parse_dates=['date'])
dfShow = pd.read_csv(filepath)
dfShow = dfShow.drop(df.columns[0], axis=1)
df = df.reset_index()['close']
data_array = scaler.fit_transform(np.array(df).reshape(-1,1))
#df_scaled = scaler.transform(df)
#data_array = df_scaled

st.title("Stock Prediction Using LSTM Model")

st.subheader("Sample of Training Data")
st.write(dfShow.head())  

st.subheader("Training and Testing Data")
train_size = int(len(data_array) * 0.65)  
train_data = data_array[:train_size]
test_data = data_array[train_size:]

train_data_plot = scaler.inverse_transform(train_data)
test_data_plot = scaler.inverse_transform(test_data)

fig, ax = plt.subplots()
ax.plot(train_data_plot, label="Training Data", color="orange")
ax.plot(np.arange(train_size, len(df)), test_data_plot, label="Testing Data", color="green")
ax.legend()

st.pyplot(fig)

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

    fig, ax = plt.subplots()
    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 131)
    
    ax.plot(day_new, scaler.inverse_transform(data_array[-100:]), label="Last 100 days")
    ax.plot(day_pred, out, label="Next 30 days prediction")
    ax.legend()
    
    st.pyplot(fig)

    predicted_values_df = pd.DataFrame(out.flatten()).T  

    st.write("Predicted values for the next 30 days:")
    st.table(predicted_values_df)
