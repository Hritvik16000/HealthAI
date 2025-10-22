from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.common.mlflow_utils import init_mlflow
import csv

def make_image(cls, seed=None):
    rng = np.random.default_rng(seed)
    img = rng.normal(0.5, 0.12, size=(128,128))
    if cls == "pneumonia":
        x0, y0 = rng.integers(32, 96, size=2)
        rr, cc = np.ogrid[:128,:128]
        mask = (rr-x0)**2 + (cc-y0)**2 <= rng.integers(8,16)**2
        img[mask] += rng.uniform(0.25, 0.45)
    img = np.clip(img, 0, 1)
    return img

def main():
    mlflow = init_mlflow()
    base = Path("data/raw/images")
    for cls in ["normal","pneumonia"]:
        (base/cls).mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for cls in ["normal","pneumonia"]:
        for i in range(400):
            img = make_image(cls, seed=1000+idx)
            fn = base/cls/f"img_{i:04d}.png"
            plt.imsave(fn, img, cmap="gray")
            rows.append([str(fn), cls])
            idx += 1
    labels_csv = Path("data/raw/image_labels.csv")
    with open(labels_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","label"])
        w.writerows(rows)
    mlflow.start_run(run_name="data_generate_images")
    mlflow.log_artifact(str(labels_csv))
    mlflow.end_run()

if __name__ == "__main__":
    main()
