# flask_app.py
from flask import Flask, request, jsonify
import joblib, os, pandas as pd

app = Flask(__name__)
MODELS_DIR = r'C:\Users\savan\OneDrive\Desktop\i\ml\models'

# cache loaded models in memory
_loaded_models = {}

def get_model(name):
    if name not in _loaded_models:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {name} not found.")
        _loaded_models[name] = joblib.load(path)
    return _loaded_models[name]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    model_name = data.pop('model_name', None)
    if not model_name:
        return jsonify({'error': "Please provide 'model_name' in payload."}), 400

    try:
        df = pd.DataFrame([data])
        model = get_model(model_name)
        pred = model.predict(df)[0]
        conf = model.predict_proba(df).max()
        return jsonify({
            'model': model_name,
            'prediction': int(pred),
            'confidence': round(float(conf), 3)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
