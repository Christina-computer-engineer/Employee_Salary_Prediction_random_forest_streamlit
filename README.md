# 🧑‍💼 Employee Salary Prediction using Random Forest & Streamlit

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
-  `runtime.txt` — (Optional) Specifies Python runtime

## ▶️ How to Run

Run the Streamlit app 

use this link:

https://christina-computer-engineer-employee-salary-predicti-app-rohmjl.streamlit.app/

Run the Streamlit app in VS Code
Use this command:

streamlit run app.py


📁 Project Structure

├── app.py
├──runtime.txt
├── salary_model.pkl
├── encoders.pkl
├── target_encoder.pkl
├── requirements.txt
└── README.md

🔍 Features

    Interactive UI using Streamlit

    Encoded input fields

    Predicts income class (>50K / <=50K)

    Visual insights (matplotlib / seaborn)

📦 Requirements

Refer to requirements.txt for all necessary Python libraries.

