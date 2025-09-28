# Load model
import joblib

def load_model(path='ml/model.pkl'):
    return joblib.load(path)