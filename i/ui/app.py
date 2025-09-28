import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os, joblib
from io import BytesIO, StringIO
import plotly.io as pio

st.set_page_config(page_title="IO Burst Model Comparison", layout="wide")
st.title("🚀 I/O Burst Prediction – Multi-Model Dashboard")

# CONFIG
DATA_PATH = r"C:\Users\savan\OneDrive\Desktop\7th pro\IO Final2\i\data\raw\system_io_log.csv"
MODELS_DIR = r"C:\Users\savan\OneDrive\Desktop\7th pro\IO Final2\i\ml\models"
PROB_THRESHOLD = 0.5

# Define only the models to compare
ALLOWED_MODELS = {
    "decision_tree": "Decision Tree",
    "knn_model": "K-Nearest Neighbors",
    "logistic_regression": "Logistic Regression",
    "random_forset": "Random Forest",
    "svm_model": "Support Vector Machine"
}

@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'total_bytes_sec'])
    df['rolling_avg'] = df['total_bytes_sec'].rolling(window=3).mean().fillna(0)
    return df

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Load data
if not os.path.exists(DATA_PATH):
    st.error("CSV file not found!")
    st.stop()

df = load_data()
X = df[['total_bytes_sec', 'rolling_avg']].astype(float)

# Collect model results
model_scores = []

st.header("📊 Model-wise Burst Prediction Graphs")

for model_file in os.listdir(MODELS_DIR):
    if model_file.endswith(".pkl"):
        raw_model_name = model_file.replace(".pkl", "")
        if raw_model_name not in ALLOWED_MODELS:
            continue  # Skip models not in the approved list

        display_name = ALLOWED_MODELS[raw_model_name]
        model_path = os.path.join(MODELS_DIR, model_file)
        model = load_model(model_path)

        # Run prediction
        df['probability'] = model.predict_proba(X)[:, 1]
        df['prediction'] = model.predict(X)

        # Accuracy
        accuracy = round((df['prediction'] == (df['probability'] >= PROB_THRESHOLD)).mean(), 4)

        # Graph Clarity Evaluation
        probs = df['probability']
        valid_df = df.iloc[5:].copy()
        valid_probs = valid_df['probability']
        burst_found = (valid_probs >= PROB_THRESHOLD)

        if burst_found.any():
            burst_idx = burst_found.idxmax()
            burst_time = df.loc[burst_idx, 'timestamp']
            time_to_burst = (burst_time - df['timestamp'].iloc[0]).total_seconds()
        else:
            time_to_burst = float('inf')

        slope = probs.diff().mean()
        if probs.std() < 0.01:
            slope = 0.0001

        clarity_score = round((1 / (time_to_burst + 1)) + (slope * 5) + accuracy, 3) if time_to_burst != float('inf') else round(slope * 5 + accuracy, 3)

        model_scores.append({
            "Model": display_name,
            "Calculated Accuracy": accuracy,
            "Time to Burst (s)": round(time_to_burst, 2) if time_to_burst != float('inf') else "N/A",
            "Avg. Prob Slope": round(slope, 4),
            "Graph Clarity Score": clarity_score
        })

        # Plot graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['total_bytes_sec'],
                                 name="I/O (bytes/sec)", yaxis="y1", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['probability'],
                                 name="Burst Probability", yaxis="y2", line=dict(dash="dash")))
        fig.update_layout(
            title=f"{display_name} - I/O vs Burst Probability",
            xaxis=dict(title="Timestamp"),
            yaxis=dict(title="I/O (bytes/sec)", side="left"),
            yaxis2=dict(title="Burst Probability", overlaying="y", side="right", range=[0, 1]),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download graph image
        img_bytes = fig.to_image(format="png")
        st.download_button(
            label=f"📥 Download Graph - {display_name}",
            data=img_bytes,
            file_name=f"{raw_model_name}_graph.png",
            mime="image/png"
        )

        # Download predictions
        pred_df = df[['timestamp', 'total_bytes_sec', 'rolling_avg', 'probability', 'prediction']]
        csv_data = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"⬇️ Download CSV - {display_name}",
            data=csv_data,
            file_name=f"{raw_model_name}_predictions.csv",
            mime="text/csv"
        )

# Comparison Table
st.header("📋 Model Accuracy & Graph Clarity Comparison")
if model_scores:
    df_scores = pd.DataFrame(model_scores)
    st.dataframe(df_scores)

    best_model_row = df_scores.loc[df_scores['Graph Clarity Score'].idxmax()]
    st.success(f"🏆 Best Performing Model (Auto‑Detected): **{best_model_row['Model']}**\n"
               f"✔️ Accuracy: **{best_model_row['Calculated Accuracy']}**\n"
               f"📉 Time to Burst: **{best_model_row['Time to Burst (s)']}s**\n"
               f"🧠 Graph Clarity Score: **{best_model_row['Graph Clarity Score']}**")

    compare_csv = df_scores.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📊 Download Comparison Table as CSV",
        compare_csv,
        "model_comparison_table.csv",
        "text/csv"
    )

    fig_table = go.Figure(data=[go.Table(
        header=dict(values=list(df_scores.columns),
                    fill_color='lightblue',
                    align='left'),
        cells=dict(values=[df_scores[col] for col in df_scores.columns],
                   fill_color='white',
                   align='left'))
    ])
    fig_table.update_layout(title="Model Comparison Table", margin=dict(l=5, r=5, t=40, b=5))
    table_img = fig_table.to_image(format="png")
    st.download_button(
        "🖼️ Download Comparison Table as PNG",
        data=table_img,
        file_name="model_comparison_table.png",
        mime="image/png"
    )
else:
    st.warning("No models found in the models directory.")
