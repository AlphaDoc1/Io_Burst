# Predict function
def predict_io_burst(model, data):
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0].max()
    return prediction, probability