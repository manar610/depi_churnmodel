import streamlit as st
import pandas as pd
import pickle

# Load model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Sidebar
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    This app predicts whether a telecom customer is likely to churn based on their service usage and account details. 
    Just fill in the information and click **Predict Churn** to get results!
    """)

# Title
st.title("📞 Customer Churn Prediction")
st.markdown("Please fill in the customer's details below:")

# Group input fields into two columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 1)

with col2:
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", [
        "Bank transfer (automatic)", 
        "Credit card (automatic)", 
        "Electronic check", 
        "Mailed check"])

# Automatically calculated
total_charges = tenure * monthly_charges

st.markdown("---")
st.subheader("🔍 Prediction Result")

# One-hot encoded fields
internet_fiber = 1 if internet_service == "Fiber optic" else 0
internet_no = 1 if internet_service == "No" else 0
contract_one_year = 1 if contract == "One year" else 0
contract_two_year = 1 if contract == "Two year" else 0
payment_credit = 1 if payment_method == "Credit card (automatic)" else 0
payment_electronic = 1 if payment_method == "Electronic check" else 0
payment_mailed = 1 if payment_method == "Mailed check" else 0

# Input dictionary
input_data = {
    'gender': 1 if gender == "Male" else 0,
    'SeniorCitizen': 1 if senior == "Yes" else 0,
    'Partner': 1 if partner == "Yes" else 0,
    'Dependents': 1 if dependents == "Yes" else 0,
    'tenure': tenure,
    'PhoneService': 1,  # Defaulted to 1
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

# Prediction
if st.button("🎯 Predict Churn"):
    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]
    if prediction == 1:
        st.error(f"⚠️ The customer is likely to churn.\n\nConfidence: **{prob:.2%}**")
    else:
        st.success(f"✅ The customer is unlikely to churn.\n\nConfidence: **{1 - prob:.2%}**")

# Footer
st.markdown("---")
st.markdown("👩‍💻 Created with ❤️ by **Manar Hisham*")
