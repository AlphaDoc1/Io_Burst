import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

df = pd.read_csv(r'C:\Users\savan\OneDrive\Desktop\i\data\processed\processed_data.csv')
X = df[['total_bytes_sec', 'rolling_avg']]
y = df['burst']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ [SVM] Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print(classification_report(y_test, y_pred))

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/svm_model.pkl')
