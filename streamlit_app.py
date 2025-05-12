import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load model
model = pickle.load(open("churn_model.pkl", "rb"))

# Load dataset for visualizations
data = pd.read_csv("cleaned_data.csv")

# App title
st.title("📊 Customer Churn Prediction")

# Sidebar for inputs
st.sidebar.header("Input Features")

# Input fields
tenure = st.sidebar.number_input("Customer Tenure", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=720.0)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# Encoding inputs as needed for the model
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "gender_Female": 1 if gender == "Female" else 0,
    "gender_Male": 1 if gender == "Male" else 0,
    "InternetService_DSL": 1 if internet_service == "DSL" else 0,
    "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
    "InternetService_No": 1 if internet_service == "No" else 0,
    "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0
}

input_df = pd.DataFrame([input_dict])

# Predict button
if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)

    # Display prediction
    if prediction[0] == 1:
        st.error("❌ Prediction: Customer is likely to churn")
    else:
        st.success("✅ Prediction: Customer is not likely to churn")

    # Feature importance and visualizations
    st.subheader("📌 Feature Importance and Insights")

    # Feature importance bar chart
    importances = model.feature_importances_
    features = input_df.columns
    indices = np.argsort(importances)

    fig_feat, ax_feat = plt.subplots()
    ax_feat.barh(range(len(indices)), importances[indices], align='center')
    ax_feat.set_yticks(range(len(indices)))
    ax_feat.set_yticklabels(features[indices])
    ax_feat.set_title('Feature Importance')
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_feat)

    # Churn rate pie chart
    churn_counts = data['Churn'].value_counts()
    labels = ['Retained', 'Churned']
    colors = ['#1f77b4', '#ff7f0e']
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(churn_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax_pie.axis('equal')
    with col2:
        st.pyplot(fig_pie)

    # Density plot for churned vs retained
    st.subheader("📈 Churned vs. Retained - Monthly Charges")
    fig_kde, ax_kde = plt.subplots()
    sns.kdeplot(data=data[data['Churn'] == 1]['MonthlyCharges'], label="Churned", shade=True, color="#ff7f0e", ax=ax_kde)
    sns.kdeplot(data=data[data['Churn'] == 0]['MonthlyCharges'], label="Retained", shade=True, color="#1f77b4", ax=ax_kde)
    ax_kde.set_xlabel("Monthly Charges")
    ax_kde.set_ylabel("Density")
    ax_kde.legend()
    st.pyplot(fig_kde)
