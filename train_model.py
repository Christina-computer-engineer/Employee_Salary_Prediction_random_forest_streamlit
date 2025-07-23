import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("adult.csv")

# Clean column names (remove whitespace)
data.columns = [col.strip() for col in data.columns]

# Drop rows with missing values ("?")
data.replace("?", pd.NA, inplace=True)
data.dropna(inplace=True)

# Encode categorical columns using original names
cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Encode target column: income
target_encoder = LabelEncoder()
data['income'] = target_encoder.fit_transform(data['income'])  # <=50K = 0, >50K = 1

# Features and target
X = data.drop('income', axis=1)
y = data['income']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
# joblib.dump(model, "salary_model.pkl")
# joblib.dump(encoders, "encoders.pkl")
# joblib.dump(target_encoder, "target_encoder.pkl")
# Save model and encoders with compression to reduce file size
joblib.dump(model, "salary_model.pkl", compress=3)
joblib.dump(encoders, "encoders.pkl", compress=3)
joblib.dump(target_encoder, "target_encoder.pkl", compress=3)

