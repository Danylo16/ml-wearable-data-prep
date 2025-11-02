# Wearable Sensor Data Pipeline

Small, focused project to simulate **wearable** sensor data (accelerometer + heart rate + skin temperature), preprocess it, and turn it into ML-ready features for activity recognition.

## What this repo does
- generates labeled time-series for `sitting`, `walking`, `running`, `stairs`
- cleans signals (gravity removal for accel)
- windows the data (e.g., 2 s, 50% overlap)
- extracts simple statistical features
- includes a notebook to visualize raw signals and inspect feature distributions

## Structure
├── README.md
├── requirements.txt
├── src/
│ ├── generate_data.py
│ ├── preprocess.py
│ └── utils.py
├── data/
│ ├── raw/ # generated CSVs (ignored)
│ └── features/ # ML-ready features (kept)
└── notebooks/
└── visualize.ipynb

## Quickstart
```bash
pip install -r requirements.txt

# 1) generate data
python src/generate_data.py --minutes 20 --fs 50 --out data/raw/session_01.csv

# 2) preprocess to features
python src/preprocess.py --csv data/raw/session_01.csv --fs 50 --win_s 2.0 --overlap 0.5 --out data/features/session_01_features.csv


```
## Data formats
Raw CSV

timestamp,ax,ay,az,heart_rate,skin_temp,activity
2025-11-01T12:00:00.000Z,-0.12,0.01,9.72,78,33.4,walking

Features CSV 

ax_mean,ax_std,...,sma,ax_ac1,ay_ac1,az_ac1,activity
-0.03,0.41,...,1.23,0.54,0.32,0.48,walking


## Notes

accel includes gravity in az; preprocessing removes it via high-pass

HR follows activity-dependent baseline with smoothed noise

windows/overlap configurable; defaults tuned for 50 Hz

features kept simple on purpose (TinyML-friendly)