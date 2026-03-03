# Customer-Churn-Predictor

# Customer Churn Predictor (ML + Streamlit)

A beginner-friendly classification project that predicts whether a customer is likely to churn.

## What this project demonstrates
- Data loading + preprocessing
- One-hot encoding categorical features
- Training a classification model (Logistic Regression)
- Saving and loading a trained model
- Deploying a small Streamlit web app

## Files
- `dataset.csv` — sample churn dataset
- `train_model.py` — trains model and saves `churn_model.joblib`
- `app.py` — Streamlit UI for churn prediction
- `requirements.txt` — dependencies

## Run
pip install -r requirements.txt  
python train_model.py  
streamlit run app.py  

## What I Learned
- How to build an end-to-end classification pipeline using scikit-learn
- How to encode categorical variables safely using OneHotEncoder
- How to evaluate a classifier using accuracy + confusion matrix + classification report
- How to package an ML model into a simple web UI using Streamlit
