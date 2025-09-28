import psutil
import csv
import time
import os
from datetime import datetime

output_dir = r'C:\Users\savan\OneDrive\Desktop\i\data\raw'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'system_io_log.csv')

# Header
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['timestamp', 'read_bytes_sec', 'write_bytes_sec', 'total_bytes_sec'])

print("⏳ Collecting system disk IO data every 1 sec... Press Ctrl+C to stop.")

try:
    prev_read = psutil.disk_io_counters().read_bytes
    prev_write = psutil.disk_io_counters().write_bytes

    while True:
        time.sleep(1)  # sample every 1 second
        current = psutil.disk_io_counters()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Calculate bytes per second
        read_per_sec = current.read_bytes - prev_read
        write_per_sec = current.write_bytes - prev_write
        total_per_sec = read_per_sec + write_per_sec

        # Save previous for next iteration
        prev_read = current.read_bytes
        prev_write = current.write_bytes

        # Write to CSV
        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, read_per_sec, write_per_sec, total_per_sec])

except KeyboardInterrupt:
    print("\n✅ Stopped. High-accuracy IO log saved at:")
    print(f"📁 {output_file}")
