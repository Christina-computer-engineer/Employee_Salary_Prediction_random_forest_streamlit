# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
# from sklearn.model_selection import train_test_split

# # Page config
# st.set_page_config(page_title="Employee Salary Prediction", layout="wide")

# # Load model and encoders
# model = joblib.load("salary_model.pkl")
# encoders = joblib.load("encoders.pkl")
# target_encoder = joblib.load("target_encoder.pkl")

# # Load and preprocess dataset for evaluation
# @st.cache_data
# def load_and_prepare_data():
#     data = pd.read_csv("adult.csv")
#     data.columns = [col.strip() for col in data.columns]
#     data.replace("?", pd.NA, inplace=True)
#     data.dropna(inplace=True)
#     cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
#     for col in cat_cols:
#         data[col] = encoders[col].transform(data[col])
#     data['income'] = target_encoder.transform(data['income'])
#     return data

# data = load_and_prepare_data()
# X = data.drop('income', axis=1)
# y = data['income']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # --- Streamlit Tabs ---
# tab1, tab2 = st.tabs(["🧠 Predict Salary", "📊 Model Evaluation"])

# # -------------------------
# # TAB 1: Salary Prediction
# # -------------------------
# with tab1:
#     st.title("🧑‍💼 Employee Salary Prediction")
#     st.markdown("Enter employee details to predict if income is `>50K` or `<=50K`")

#     # Input form
#     with st.form("prediction_form"):
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             age = st.number_input("Age", min_value=18, max_value=100, value=30)
#             education = st.selectbox("Education", encoders['education'].classes_)
#             education_num = st.slider("Education Num", 1, 16, 10)
#         with col2:
#             workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
#             marital_status = st.selectbox("Marital Status", encoders['marital-status'].classes_)
#             occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
#         with col3:
#             relationship = st.selectbox("Relationship", encoders['relationship'].classes_)
#             race = st.selectbox("Race", encoders['race'].classes_)
#             gender = st.selectbox("Gender", encoders['gender'].classes_)
        
#         capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
#         capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
#         hours_per_week = st.slider("Hours per Week", 1, 100, 40)
#         native_country = st.selectbox("Native Country", encoders['native-country'].classes_)

#         submitted = st.form_submit_button("Predict")

#     if submitted:
#         input_data = {
#             'age': age,
#             'workclass': encoders['workclass'].transform([workclass])[0],
#             'fnlwgt': 200000,  # Fixed placeholder value
#             'education': encoders['education'].transform([education])[0],
#             'educational-num': education_num,
#             'marital-status': encoders['marital-status'].transform([marital_status])[0],
#             'occupation': encoders['occupation'].transform([occupation])[0],
#             'relationship': encoders['relationship'].transform([relationship])[0],
#             'race': encoders['race'].transform([race])[0],
#             'gender': encoders['gender'].transform([gender])[0],
#             'capital-gain': capital_gain,
#             'capital-loss': capital_loss,
#             'hours-per-week': hours_per_week,
#             'native-country': encoders['native-country'].transform([native_country])[0]
#         }

#         input_df = pd.DataFrame([input_data])
#         pred = model.predict(input_df)[0]
#         label = target_encoder.inverse_transform([pred])[0]

#         if pred == 1:
#             st.success(f"✅ Predicted Salary: {label}")
#         else:
#             st.warning(f"❌ Predicted Salary: {label}")

# # -------------------------
# # TAB 2: Model Evaluation
# # -------------------------
# with tab2:
#     st.title("📊 Model Evaluation Dashboard")

#     # Predictions
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:, 1]

#     # Confusion Matrix
#     st.subheader("🔲 Confusion Matrix")
#     cm = confusion_matrix(y_test, y_pred)
#     fig1, ax1 = plt.subplots()
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
#                 xticklabels=target_encoder.classes_,
#                 yticklabels=target_encoder.classes_)
#     ax1.set_title("Confusion Matrix")
#     ax1.set_xlabel("Predicted")
#     ax1.set_ylabel("Actual")
#     st.pyplot(fig1)

#     # Classification Report
#     st.subheader("📋 Classification Report")
#     report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)
#     report_df = pd.DataFrame(report).transpose()
#     st.dataframe(report_df.style.highlight_max(axis=0))

#     # ROC Curve
#     st.subheader("📈 ROC Curve")
#     fpr, tpr, _ = roc_curve(y_test, y_proba)
#     auc_val = roc_auc_score(y_test, y_proba)
#     fig2, ax2 = plt.subplots()
#     ax2.plot(fpr, tpr, label=f"AUC = {auc_val:.2f}", color='darkorange')
#     ax2.plot([0, 1], [0, 1], 'k--', label="Random Guess")
#     ax2.set_xlabel("False Positive Rate")
#     ax2.set_ylabel("True Positive Rate")
#     ax2.set_title("ROC Curve")
#     ax2.legend(loc="lower right")
#     st.pyplot(fig2)

#     # Feature Importances
#     st.subheader("📊 Feature Importances")
#     importances = model.feature_importances_
#     feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
#     feat_df = feat_df.sort_values(by="Importance", ascending=False)

#     fig3, ax3 = plt.subplots(figsize=(10, 6))
#     sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis", ax=ax3)
#     ax3.set_title("Feature Importance in Salary Prediction")
#     st.pyplot(fig3)

#     st.caption("📌 Model evaluated on test data from UCI Adult Dataset.")


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(page_title="Employee Salary Prediction", layout="wide")

# Load Model and Encoders
model = joblib.load("salary_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Load and preprocess dataset
@st.cache_data
def load_and_prepare_data():
    data = pd.read_csv("adult.csv")
    data.columns = [col.strip() for col in data.columns]
    data.replace("?", pd.NA, inplace=True)
    data.dropna(inplace=True)

    # Encode categorical columns
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    for col in cat_cols:
        data[col] = encoders[col].transform(data[col])
    data['income'] = target_encoder.transform(data['income'])
    return data

data = load_and_prepare_data()
X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit Tabs
tab1, tab2 = st.tabs(["🧠 Predict Salary", "📊 Model Evaluation"])

# ------------------------------
# TAB 1: Employee Salary Prediction
# ------------------------------
with tab1:
    st.title("🧑‍💼 Employee Salary Prediction")
    st.markdown("Fill out the form below to predict whether the employee's salary is `>50K` or `<=50K`")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            education = st.selectbox("Education", encoders['education'].classes_)
            education_num = st.slider("Education Num", 1, 16, 10)

        with col2:
            workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
            marital_status = st.selectbox("Marital Status", encoders['marital-status'].classes_)
            occupation = st.selectbox("Occupation", encoders['occupation'].classes_)

        with col3:
            relationship = st.selectbox("Relationship", encoders['relationship'].classes_)
            race = st.selectbox("Race", encoders['race'].classes_)
            gender = st.selectbox("Gender", encoders['gender'].classes_)

        capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
        hours_per_week = st.slider("Hours per Week", 1, 100, 40)
        native_country = st.selectbox("Native Country", encoders['native-country'].classes_)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = {
            'age': age,
            'workclass': encoders['workclass'].transform([workclass])[0],
            'fnlwgt': 200000,  # Static dummy value
            'education': encoders['education'].transform([education])[0],
            'educational-num': education_num,
            'marital-status': encoders['marital-status'].transform([marital_status])[0],
            'occupation': encoders['occupation'].transform([occupation])[0],
            'relationship': encoders['relationship'].transform([relationship])[0],
            'race': encoders['race'].transform([race])[0],
            'gender': encoders['gender'].transform([gender])[0],
            'capital-gain': capital_gain,
            'capital-loss': capital_loss,
            'hours-per-week': hours_per_week,
            'native-country': encoders['native-country'].transform([native_country])[0]
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        label = target_encoder.inverse_transform([prediction])[0]

        if prediction == 1:
            st.success(f"✅ Predicted Salary: {label}")
        else:
            st.warning(f"❌ Predicted Salary: {label}")

# ------------------------------
# TAB 2: Model Evaluation
# ------------------------------
with tab2:
    st.title("📊 Model Evaluation Dashboard")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    st.subheader("🔲 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=target_encoder.classes_,
                yticklabels=target_encoder.classes_)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix")
    st.pyplot(fig1)

    # Classification Report
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0))

    # ROC Curve
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Receiver Operating Characteristic")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

    # Feature Importance
    st.subheader("📊 Feature Importances")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis", ax=ax3)
    ax3.set_title("Top Features for Income Prediction")
    st.pyplot(fig3)

    st.caption("📌 This model was trained and evaluated using the UCI Adult Income Dataset.")



