import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.write("Enter customer details and predict whether the customer is likely to churn.")

@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")

model = None
try:
    model = load_model()
except Exception:
    st.warning("Model not found. Run `python train_model.py` first to create churn_model.joblib")

with st.form("churn_form"):
    tenure_months = st.number_input("Tenure (months)", 0, 120, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 50000.0, 800.0)

    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])

    num_products = st.number_input("Number of Products", 1, 10, 2)
    is_senior = st.selectbox("Senior Citizen?", [0, 1])
    has_partner = st.selectbox("Has Partner?", [0, 1])

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    if model is None:
        st.error("Please train the model first: run `python train_model.py`")
    else:
        X_new = pd.DataFrame([{
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "num_products": num_products,
            "is_senior": is_senior,
            "has_partner": has_partner
        }])

        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0][1]

        if pred == 1:
            st.error(f"Prediction: CHURN (probability {proba:.2%})")
        else:
            st.success(f"Prediction: NOT CHURN (churn probability {proba:.2%})")
