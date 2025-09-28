import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load preprocessed data
data_path = r'C:\Users\savan\OneDrive\Desktop\i\data\processed\processed_data.csv'
df = pd.read_csv(data_path)

# Features and target
X = df[['total_bytes_sec', 'rolling_avg']]
y = df['burst']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", round(acc * 100, 2), "%")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/random_forset.pkl')

