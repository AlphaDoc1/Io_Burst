# 🔥 Real-Time I/O Burst Prediction using AI/ML

This project predicts upcoming I/O (Input/Output) bursts in a computing system using machine learning. It processes real-time system data, trains a model, and visualizes burst predictions live on an interactive dashboard. It’s designed for both testing and real-world deployment, making it suitable for academic and practical use.

---

## ✅ What the System Does

- Collects real-time system I/O data using `psutil`.
- Preprocesses the data using rolling averages and burst detection logic.
- Trains a Random Forest Classifier using scikit-learn.
- Predicts bursts with confidence and estimates future burst times.
- Visualizes everything live using a beautiful Streamlit dashboard.
- Includes download options (CSV, PNG) for results and graph data.

---

## 📁 Project Directory Structure

```
IOBurstPredictor/
├── backend/         # Flask API (model inference)
├── ml/              # Preprocessing and model training
├── ui/              # Streamlit Dashboard
├── realtime/        # Real-time data collector
├── data/
│   ├── raw/         # Live I/O logs
│   └── processed/   # Preprocessed data
```

---

## 🛠️ Technologies Used

| Component        | Technology           |
|------------------|----------------------|
| Machine Learning | Random Forest (scikit-learn)  
| Data Collection  | psutil  
| Preprocessing    | pandas  
| Backend API      | Flask  
| Visualization    | Streamlit + Plotly  
| Language         | Python 3.x  
| Format           | CSV  

---

## ⚙️ How to Run the Entire Project (Step-by-Step)

### 🔧 1. Install Required Dependencies

```bash
pip install flask streamlit pandas scikit-learn plotly psutil joblib
```

---

### 📥 2. Collect Real-Time System I/O Data

```bash
cd realtime
python system_data_collector.py
```

➡️ Let it run for a few minutes and press `Ctrl + C` to stop.  
➡️ The raw data will be saved in `data/raw/system_io_log.csv`.

---

### 🧹 3. Preprocess the Collected Data

```bash
cd ../ml
python preprocess.py
```

➡️ This generates `data/processed/processed_data.csv` with rolling averages and burst labels.

---

### 🧠 4. Train the ML Model

```bash
python train_model.py
```

➡️ Trains a Random Forest model and saves it as `model.pkl`.

---

### 🚀 5. Start the Flask Prediction API

```bash
cd ../backend
python app.py
```

➡️ The model is served at `http://localhost:5000`.

---

### 📊 6. Launch the Streamlit Dashboard

```bash
cd ../ui
streamlit run app.py
```

➡️ Visit: [http://localhost:8501](http://localhost:8501) to view live predictions.

---

## 🔍 Live Dashboard Includes

- 🔴 Live I/O Graph (last 50 points)
- ⚠️ Current Burst Status + Confidence
- ⏳ Estimated Time of Next Burst
- 📥 Download CSV and PNG options

---

## 🔬 Example Prediction Output

```
📥 Current I/O: 2,000,000 bytes/sec  
📊 Rolling Avg: 1,166,666  
⚠️ BURST Detected! (Confidence: 99.00%)  
📍 Status: 🔴 Currently in Burst at 2025-07-21 10:51:15  
🕓 Estimated Next Burst: 2025-07-21 10:54:05  
```

---

## 🧪 Sample Test Data

If you don’t want to wait for live data collection, use the sample file below to simulate testing:

```
data/raw/system_io_log.csv
```

---

## 🏁 Final Notes

- The system is designed to predict **future I/O bursts** based on real patterns.
- You can tune the model by adjusting the rolling window or training data volume.
- Ideal for research, benchmarking, and academic paper submissions.

---

> 🔒 Fully customizable and extensible for production use.  
> 📄 Ready to publish for academic or industry research.
