
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load LSTM Model
model = tf.keras.models.load_model("Notebook/lstm_model_11-06-2026-06-06-59.keras")

st.title("Rossmann Sales Prediction Dashboard")

uploaded_file = st.file_uploader(
    "Upload CSV file (must contain Sales column)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data")
    st.dataframe(df.head())

    if "Sales" in df.columns:

        sales = df["Sales"].values

        if len(sales) >= 7:

            predictions = []

            for i in range(len(sales) - 7):

                sequence = sales[i:i+7]
                sequence = sequence.reshape((1, 7, 1))

                pred = model.predict(sequence, verbose=0)

                predictions.append(pred[0][0])

            result = pd.DataFrame({
                "Actual Sales": sales[7:],
                "Predicted Sales": predictions
            })

            st.subheader("Predictions")
            st.dataframe(result)

            fig, ax = plt.subplots()

            ax.plot(result["Actual Sales"], label="Actual")
            ax.plot(result["Predicted Sales"], label="Predicted")

            ax.legend()

            st.pyplot(fig)

            csv = result.to_csv(index=False)

            st.download_button(
                "Download Predictions",
                csv,
                "predictions.csv",
                "text/csv"
            )

        else:
            st.error("CSV must contain at least 7 Sales values.")

    else:
        st.error("Sales column not found.")
