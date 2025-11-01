 
import argparse, os, numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

def butter_highpass(cut, fs, order=3):
    b,a = butter(order, cut/(0.5*fs), btype='high', analog=False)
    return b,a

def filt_highpass(x, fs, cut=0.25):
    b,a = butter_highpass(cut, fs)
    return filtfilt(b, a, x)

def magnitude(ax, ay, az):
    return np.sqrt(ax**2 + ay**2 + az**2)

def window_indices(n, win, step):
    starts = np.arange(0, n-win+1, step, dtype=int)
    return [(s, s+win) for s in starts]

def features_from_window(dfw):
    ax, ay, az = dfw["ax"].values, dfw["ay"].values, dfw["az"].values
    m = magnitude(ax, ay, az)
    hr = dfw["heart_rate"].values
    tmp = dfw["skin_temp"].values

    def stats(x):
        return [np.mean(x), np.std(x), np.min(x), np.max(x)]
    feat = []
    for arr in [ax, ay, az, m, hr, tmp]:
        feat += stats(arr)

    sma = (np.mean(np.abs(ax)) + np.mean(np.abs(ay)) + np.mean(np.abs(az)))
    def ac1(x):
        x = x - x.mean()
        return np.correlate(x[:-1], x[1:])[0] / (np.sum(x**2)+1e-9)
    feat += [sma, ac1(ax), ac1(ay), ac1(az)]
    return np.array(feat, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fs", type=int, default=50)
    ap.add_argument("--win_s", type=float, default=2.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--out", default="data/features/features.csv")
    ap.add_argument("--standardize", action="store_true", help="fit+apply scaler here (default: off)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for ax_name in ["ax","ay","az"]:
        df[ax_name] = filt_highpass(df[ax_name].values, fs=args.fs, cut=0.25)

    win = int(args.win_s * args.fs)
    step = int(win * (1.0 - args.overlap))
    idx = window_indices(len(df), win, step)

    rows = []
    wid = 0
    for s,e in idx:
        dfw = df.iloc[s:e]
        X = features_from_window(dfw)
        label = dfw["activity"].mode()[0]
        rows.append(np.concatenate([X, np.array([label, wid], dtype=object)]))
        wid += 1

    base = ["ax","ay","az","mag","hr","temp"]
    cols = []
    for name in base:
        cols += [f"{name}_mean", f"{name}_std", f"{name}_min", f"{name}_max"]
    cols += ["sma", "ax_ac1", "ay_ac1", "az_ac1", "activity", "window_id"]

    feat_df = pd.DataFrame(rows, columns=cols)

    if args.standardize:
        scaler = StandardScaler()
        feat_cols = [c for c in feat_df.columns if c not in ("activity","window_id")]
        feat_df[feat_cols] = scaler.fit_transform(feat_df[feat_cols])

    feat_df.to_csv(args.out, index=False)
    print(f"Wrote {len(feat_df)} windows to {args.out}")

if __name__ == "__main__":
    main()
 