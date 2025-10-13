from __future__ import annotations
import csv, math
import numpy as np


def gen(n=5000, seed=7):
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.float32)
    R = np.zeros(n, dtype=np.float32)
    kappa = np.zeros(n, dtype=np.float32)
    H = np.zeros(n, dtype=np.float32)
    # regime 1 -> 2 with higher vol and sync
    for t in range(n):
        if t < n // 2:
            x[t] = rng.normal(0, 0.001)
            R[t] = 0.5 + 0.05 * rng.random()
            kappa[t] = 0.1 + 0.05 * rng.normal()
        else:
            x[t] = rng.normal(0, 0.004) + (0.02 if t % 100 < 2 else 0.0)
            R[t] = 0.65 + 0.05 * rng.random()
            kappa[t] = -0.1 + 0.05 * rng.normal()
        H[t] = max(0.0, abs(x[t]) * 150)
    return x, R, kappa, H


def write_csv(path="/mnt/data/amm_synth.csv"):
    x, R, kappa, H = gen()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "R", "kappa", "H"])
        for i in range(len(x)):
            w.writerow([float(x[i]), float(R[i]), float(kappa[i]), float(H[i])])
    return path


if __name__ == "__main__":
    print(write_csv())
