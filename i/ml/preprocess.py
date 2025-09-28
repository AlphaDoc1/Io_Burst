import pandas as pd
import os

# Load real data
input_file = r'C:\Users\savan\OneDrive\Desktop\7th pro\IO Final2\i\data\raw\system_io_log.csv'
df = pd.read_csv(input_file)

# Compute rolling average (over 3 seconds)
df['rolling_avg'] = df['total_bytes_sec'].rolling(window=3).mean().fillna(0)

# Label burst: > 1MB/s (1,000,000 bytes/sec)
df['burst'] = df['total_bytes_sec'].apply(lambda x: 1 if x > 1000000 else 0)

# Save processed file
output_path = r'C:\Users\savan\OneDrive\Desktop\7th pro\IO Final2\i\data\processed'
os.makedirs(output_path, exist_ok=True)
df[['total_bytes_sec', 'rolling_avg', 'burst']].to_csv(
    os.path.join(output_path, 'processed_data.csv'), index=False
)

print("✅ Preprocessed data saved to:", os.path.join(output_path, 'processed_data.csv'))
