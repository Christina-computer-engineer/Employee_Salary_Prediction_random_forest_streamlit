# 🧑‍💼 Employee Salary Prediction using Streamlit

This project predicts whether a person earns more than 50K per year based on demographic features using a Random Forest Classifier. It is deployed with Streamlit and built in Python.

## 🚀 Features

- Predicts income class (<=50K or >50K)
- Inputs include age, education, workclass, gender, hours/week, etc.
- Visualizations (if implemented): Feature Importance, Confusion Matrix
- Trained model saved as `.pkl` and used for predictions

## 🧠 Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- Seaborn / Matplotlib (optional)

## 📁 Project Files

- `app.py` — Streamlit app
- `adult.csv` — Dataset
- `salary_model.pkl` — Trained Random Forest model
- `encoders.pkl` — Categorical feature encoders
- `target_encoder.pkl` — Target label encoder
- `requirements.txt` — All Python dependencies

## ▶️ How to Run

Run the Streamlit app in VS Code
Use this command:

streamlit run app.py
