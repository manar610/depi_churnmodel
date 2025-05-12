import streamlit as st
import pandas as pd
import pickle

# Load model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction")
st.write("Fill the information below to predict whether a customer is likely to churn.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72, 1)
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = tenure * monthly_charges


# Multi-class: InternetService
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
internet_fiber = 1 if internet_service == "Fiber optic" else 0
internet_no = 1 if internet_service == "No" else 0

# Multi-class: Contract
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
contract_one_year = 1 if contract == "One year" else 0
contract_two_year = 1 if contract == "Two year" else 0

# Multi-class: Payment Method
payment_method = st.selectbox("Payment Method", [
    "Bank transfer (automatic)", 
    "Credit card (automatic)", 
    "Electronic check", 
    "Mailed check"])
payment_credit = 1 if payment_method == "Credit card (automatic)" else 0
payment_electronic = 1 if payment_method == "Electronic check" else 0
payment_mailed = 1 if payment_method == "Mailed check" else 0

# Input dictionary with all required features
input_data = {
    'gender': 1 if gender == "Male" else 0,
    'SeniorCitizen': 1 if senior == "Yes" else 0,
    'Partner': 1 if partner == "Yes" else 0,
    'Dependents': 1 if dependents == "Yes" else 0,
    'tenure': tenure,
    'PhoneService': 1,  # assumed default
    'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'MultipleLines_No phone service': 0,
    'MultipleLines_Yes': 0,
    'InternetService_Fiber optic': internet_fiber,
    'InternetService_No': internet_no,
    'OnlineSecurity_No internet service': 1 if internet_service == "No" else 0,
    'OnlineSecurity_Yes': 0,
    'OnlineBackup_No internet service': 1 if internet_service == "No" else 0,
    'OnlineBackup_Yes': 0,
    'DeviceProtection_No internet service': 1 if internet_service == "No" else 0,
    'DeviceProtection_Yes': 0,
    'TechSupport_No internet service': 1 if internet_service == "No" else 0,
    'TechSupport_Yes': 0,
    'StreamingTV_No internet service': 1 if internet_service == "No" else 0,
    'StreamingTV_Yes': 0,
    'StreamingMovies_No internet service': 1 if internet_service == "No" else 0,
    'StreamingMovies_Yes': 0,
    'Contract_One year': contract_one_year,
    'Contract_Two year': contract_two_year,
    'PaymentMethod_Credit card (automatic)': payment_credit,
    'PaymentMethod_Electronic check': payment_electronic,
    'PaymentMethod_Mailed check': payment_mailed
}

# Convert to DataFrame
df_input = pd.DataFrame([input_data])

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]
    if prediction == 1:
        st.error(f"⚠️ Likely to Churn (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Unlikely to Churn (Confidence: {1 - prob:.2%})")
