 
import argparse, os, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta

rng = np.random.default_rng()

ACTIVITIES = [
    ("sitting", 60, 80),
    ("walking", 85, 110),
    ("running", 120, 165),
    ("stairs", 100, 140),
]

def simulate_segment(activity, seconds, fs, seed=None):
    if seed is not None:
        global rng; rng = np.random.default_rng(seed)

    n = seconds * fs
    t = np.arange(n) / fs

    noise = rng.normal(0, 0.08, size=(n, 3))
    g = np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.05, size=3)

    if activity == "sitting":
        f = 0.2; amp = 0.02
    elif activity == "walking":
        f = 1.8 + rng.normal(0,0.1); amp = 0.6
    elif activity == "running":
        f = 2.8 + rng.normal(0,0.2); amp = 1.4
    elif activity == "stairs":
        f = 2.2 + rng.normal(0,0.15); amp = 1.0
    else:
        f = 1.0; amp = 0.5

    ax = amp*np.sin(2*np.pi*f*t + rng.uniform(0, 2*np.pi)) + noise[:,0]
    ay = 0.7*amp*np.sin(2*np.pi*f*t + rng.uniform(0, 2*np.pi)) + noise[:,1]
    az = 0.3*amp*np.sin(4*np.pi*f*t + rng.uniform(0, 2*np.pi)) + noise[:,2] + g[2]

    accel = np.stack([ax, ay, az], axis=1)

    hr_min, hr_max = next(v[1:] for v in ACTIVITIES if v[0]==activity)
    hr_base = rng.uniform(hr_min, hr_max)
    hr = hr_base + np.convolve(rng.normal(0, 1, len(t)), np.ones(25)/25, mode="same")
    hr = np.clip(hr, hr_min, hr_max)

    temp_base = 33.2 + (0.6 if activity in ["running","stairs"] else 0.0)
    drift = np.cumsum(rng.normal(0, 0.002, len(t)))
    temp = temp_base + drift

    return accel, hr, temp

def generate_protocol(total_minutes=10, fs=50, seed=42):
    rng = np.random.default_rng(seed)
    segments = []
    remain = total_minutes*60
    choices = ["sitting","walking","running","stairs"]
    while remain > 0:
        act = rng.choice(choices, p=[0.25,0.35,0.25,0.15])
        sec = int(min(remain, rng.integers(90, 180)))
        segments.append((act, sec))
        remain -= sec
    # ensure all activities appear
    present = {a for a,_ in segments}
    for a in {"sitting","walking","running","stairs"} - present:
        segments.append((a, 100))
    return segments

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=10)
    ap.add_argument("--fs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/raw/session.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    protocol = generate_protocol(args.minutes, args.fs, args.seed)

    rows = []
    t0 = datetime.now(timezone.utc)
    for act, sec in protocol:
        accel, hr, temp = simulate_segment(act, sec, args.fs, seed=args.seed)
        n = sec*args.fs
        for i in range(n):
            ts = t0 + timedelta(seconds=i/args.fs)
            rows.append([ts.isoformat(timespec="milliseconds"),
                         accel[i,0], accel[i,1], accel[i,2],
                         int(hr[i]), round(float(temp[i]), 3), act])
        t0 += timedelta(seconds=sec)

    df = pd.DataFrame(rows, columns=["timestamp","ax","ay","az","heart_rate","skin_temp","activity"])
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")

if __name__ == "__main__":
    main()
 