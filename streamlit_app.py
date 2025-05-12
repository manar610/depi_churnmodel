import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(layout="wide")

# Load model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)
    
# Load dataset for visualizations
data = pd.read_csv("cleaned_data.csv")

st.title("📊 Customer Churn Prediction")

# Split into two columns
left_col, right_col = st.columns([1, 1])

# Left side: Inputs (nested columns)
with left_col:
    st.header("🔧 Input Features")
    col1, col2 = st.columns([1, 1])
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

    total_charges = tenure * monthly_charges
    

# Right side: Prediction & Visuals
with right_col:
    st.header("📈 Prediction Output")
   

    # Create input dictionary (same as your earlier structure)
    input_data = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'MultipleLines_No phone service': 0,
        'MultipleLines_Yes': 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
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
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0
    }

    df_input = pd.DataFrame([input_data])

    if st.button("🚀 Predict Churn"):
        prediction = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]
        
        if prediction == 1:
            st.error(f"⚠️ Likely to Churn (Confidence: {prob:.2%})")
        else:
            st.success(f"✅ Unlikely to Churn (Confidence: {1 - prob:.2%})")

        # Example plot (replace with your visualizations)
        st.subheader("📊 Monthly Charges Distribution")
        fig, ax = plt.subplots()
        ax.hist(df_input['MonthlyCharges'], bins=10, color="skyblue")
        ax.set_xlabel("Monthly Charges")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)


    # Feature importance and visualizations
    st.subheader("📌 Feature Importance and Insights")

    # Feature importance bar chart
    importances = model.feature_importances_
    features = df_input.columns
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
